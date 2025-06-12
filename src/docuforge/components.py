"""Core components for the DocuForge rewrite engine."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from requests.exceptions import ConnectionError, Timeout, RequestException

from .callbacks import DefaultCallbackHandler, ProgressCallbackHandler
from .models import (
    ClarificationItem,
    DocumentSection,
    DocumentStructure,
    RewriteRequest,
)
from .prompts import PromptManager, PromptType


class ComponentBase(ABC):
    """Base class for all rewrite engine components."""
    
    def __init__(
        self, 
        llm: BaseLLM,
        callback_handler: Optional[ProgressCallbackHandler] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        self.llm = llm
        self.callback_handler = callback_handler or DefaultCallbackHandler()
        self.prompt_manager = prompt_manager or PromptManager()
    
    def _invoke_llm_with_retry(self, messages: List, max_retries: int = 3, base_delay: float = 1.0):
        """Invoke LLM with retry mechanism for handling connection errors.
        
        Args:
            messages: Messages to send to LLM
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (exponential backoff)
            
        Returns:
            LLM response
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                return response
            except (ConnectionError, Timeout, RequestException) as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    error_type = type(e).__name__
                    self.callback_handler.on_stage_progress(
                        "llm_retry",
                        f"{error_type} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    error_type = type(e).__name__
                    self.callback_handler.on_stage_progress(
                        "llm_retry",
                        f"All retry attempts exhausted. Final {error_type}: {str(e)}"
                    )
            except Exception as e:
                # For non-network errors, provide detailed error info and don't retry
                error_type = type(e).__name__
                self.callback_handler.on_stage_progress(
                    "llm_error",
                    f"Non-retryable {error_type}: {str(e)}"
                )
                raise e
        
        # If we get here, all retries failed
        raise Exception(f"LLM invocation failed after {max_retries + 1} attempts: {str(last_exception)}")


class ContextBuilder(ComponentBase):
    """Builds unified context from original content and clarifications."""
    
    def build_context(self, request: RewriteRequest) -> str:
        """Build unified context string from request data.
        
        Args:
            request: The rewrite request containing original content and clarifications
            
        Returns:
            Unified context string combining original content and clarifications
        """
        context_parts = [
            "=== 原始文档内容 ===",
            request.original_content.strip(),
            ""
        ]
        
        if request.clarifications:
            context_parts.extend([
                "=== 澄清信息 ===",
                ""
            ])
            
            for i, clarification in enumerate(request.clarifications, 1):
                context_parts.extend([
                    f"问题 {i}: {clarification.question}",
                    f"回答 {i}: {clarification.answer}",
                    ""
                ])
        
        return "\n".join(context_parts).strip()
    
    def validate_context(self, context: str) -> bool:
        """Validate that the context is not empty and has reasonable length.
        
        Args:
            context: The context string to validate
            
        Returns:
            True if context is valid, False otherwise
        """
        if not context or not context.strip():
            return False
        
        # Check minimum length (should have some substantial content)
        if len(context.strip()) < 10:
            return False
        
        # Check maximum length (prevent extremely large contexts)
        if len(context) > 100000:  # 100k characters limit
            return False
        
        return True


class RevisionIssue(BaseModel):
    """Represents an issue found during document revision."""
    
    section_order: int = Field(description="Order of the section with the issue")
    issue_type: str = Field(description="Type of issue (inconsistency, missing_info, etc.)")
    description: str = Field(description="Detailed description of the issue")
    suggestion: str = Field(description="Suggested fix for the issue")


class RevisionReport(BaseModel):
    """Report containing all issues found during document revision."""
    
    issues: List[RevisionIssue] = Field(default_factory=list)
    overall_quality: str = Field(description="Overall quality assessment")
    
    @property
    def has_issues(self) -> bool:
        """Check if the report contains any issues."""
        return len(self.issues) > 0


class OutlineGenerator(ComponentBase):
    """Generates document outline and structure (Stage 1)."""
    
    def __init__(
        self, 
        llm: BaseLLM,
        callback_handler: Optional[ProgressCallbackHandler] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        super().__init__(llm, callback_handler, prompt_manager)
        self.parser = PydanticOutputParser(pydantic_object=DocumentStructure)
    
    def generate_outline(self, context: str) -> DocumentStructure:
        """Generate document outline from context.
        
        Args:
            context: Unified context containing original content and clarifications
            
        Returns:
            DocumentStructure with title and sections (content empty, goals filled)
        """
        self.callback_handler.on_stage_progress(
            "outline_generation",
            "Analyzing context and generating document structure..."
        )
        
        system_prompt, human_prompt = self.prompt_manager.build_outline_prompts(context)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        self.callback_handler.on_stage_progress(
            "outline_generation",
            "Calling LLM to generate outline..."
        )
        
        response = self._invoke_llm_with_retry(messages)
        
        try:
            outline = self.parser.parse(response.content)
            self.callback_handler.on_stage_progress(
                "outline_generation",
                f"Generated outline with {len(outline.sections)} sections"
            )
            return outline
        except Exception as e:
            self.callback_handler.on_stage_progress(
                "outline_generation",
                f"Failed to parse outline: {e}"
            )
            raise ValueError(f"Failed to parse outline from LLM response: {e}")


class ContentFiller(ComponentBase):
    """Fills content for document sections sequentially (Stage 2)."""
    
    def fill_content(
        self, 
        document: DocumentStructure, 
        context: str
    ) -> DocumentStructure:
        """Fill content for all sections in the document.
        
        Args:
            document: Document structure with empty content sections
            context: Original context for reference
            
        Returns:
            DocumentStructure with all content filled
        """
        self.callback_handler.on_stage_progress(
            "content_filling",
            f"Starting to fill content for {len(document.sections)} sections"
        )
        
        filled_document = document.model_copy(deep=True)
        
        for i, section in enumerate(filled_document.sections):
            self.callback_handler.on_stage_progress(
                "content_filling",
                f"Filling section {i+1}/{len(filled_document.sections)}: {section.title}"
            )
            
            # Build context with previous sections
            previous_content = self._build_previous_content(filled_document, i)
            
            # Fill this section
            filled_content = self._fill_section_content(
                section, context, previous_content, filled_document.title
            )
            
            section.content = filled_content
        
        self.callback_handler.on_stage_progress(
            "content_filling",
            "All sections filled successfully"
        )
        
        return filled_document
    
    def _build_previous_content(self, document: DocumentStructure, current_index: int) -> str:
        """Build content from previous sections for context continuity."""
        if current_index == 0:
            return ""
        
        previous_parts = [f"# {document.title}", ""]
        
        for section in document.sections[:current_index]:
            if section.content:
                header_prefix = "#" * (section.level + 1)
                previous_parts.extend([
                    f"{header_prefix} {section.title}",
                    "",
                    section.content,
                    ""
                ])
        
        return "\n".join(previous_parts).strip()
    
    def _fill_section_content(
        self, 
        section: DocumentSection, 
        original_context: str,
        previous_content: str,
        document_title: str
    ) -> str:
        """Fill content for a single section."""
        system_prompt, human_prompt = self.prompt_manager.build_content_prompts(
            section=section,
            original_context=original_context,
            previous_content=previous_content,
            document_title=document_title
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self._invoke_llm_with_retry(messages)
        return response.content.strip()


class Reviser(ComponentBase):
    """Reviews and revises document content (Stage 3)."""
    
    def __init__(
        self, 
        llm: BaseLLM,
        callback_handler: Optional[ProgressCallbackHandler] = None,
        prompt_manager: Optional[PromptManager] = None,
        max_revision_rounds: int = 3
    ):
        super().__init__(llm, callback_handler, prompt_manager)
        self.max_revision_rounds = max_revision_rounds
        self.parser = PydanticOutputParser(pydantic_object=RevisionReport)
    
    def review_and_revise(
        self,
        document: DocumentStructure,
        original_context: str
    ) -> DocumentStructure:
        """Review document and apply revisions if needed.
        
        Args:
            document: Complete document to review
            original_context: Original context for comparison
            
        Returns:
            Revised document structure
        """
        self.callback_handler.on_stage_progress(
            "review_revision",
            "Starting document review process"
        )
        
        current_document = document.model_copy(deep=True)
        
        for round_num in range(self.max_revision_rounds):
            self.callback_handler.on_stage_progress(
                "review_revision",
                f"Review round {round_num + 1}/{self.max_revision_rounds}"
            )
            
            # Review the document
            report = self._review_document(current_document, original_context)
            
            if not report.has_issues:
                self.callback_handler.on_stage_progress(
                    "review_revision",
                    "No issues found, document review completed"
                )
                break
            
            self.callback_handler.on_stage_progress(
                "review_revision",
                f"Found {len(report.issues)} issues, applying revisions"
            )
            
            # Apply revisions
            current_document = self._apply_revisions(
                current_document, report, original_context
            )
        
        return current_document
    
    def _review_document(
        self, 
        document: DocumentStructure, 
        original_context: str
    ) -> RevisionReport:
        """Review document and identify issues."""
        document_content = document.to_markdown()
        
        system_prompt, human_prompt = self.prompt_manager.build_review_prompts(
            document_content=document_content,
            original_context=original_context
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self._invoke_llm_with_retry(messages)
        
        try:
            return self.parser.parse(response.content)
        except Exception as e:
            # If parsing fails, assume no issues
            self.callback_handler.on_stage_progress(
                "review_revision",
                f"Failed to parse review report, assuming no issues: {e}"
            )
            return RevisionReport(issues=[], overall_quality="Unable to assess")
    
    def _apply_revisions(
        self,
        document: DocumentStructure,
        report: RevisionReport,
        original_context: str
    ) -> DocumentStructure:
        """Apply revisions based on the review report."""
        revised_document = document.model_copy(deep=True)
        
        for issue in report.issues:
            section = revised_document.get_section_by_order(issue.section_order)
            if section:
                self.callback_handler.on_stage_progress(
                    "review_revision",
                    f"Revising section: {section.title}"
                )
                
                revised_content = self._revise_section_content(
                    section, issue, original_context
                )
                section.content = revised_content
        
        return revised_document
    
    def _revise_section_content(
        self,
        section: DocumentSection,
        issue: RevisionIssue,
        original_context: str
    ) -> str:
        """Revise content for a specific section based on identified issue."""
        system_prompt, human_prompt = self.prompt_manager.build_revision_prompts(
            section=section,
            issue_description=issue.description,
            issue_suggestion=issue.suggestion,
            original_context=original_context
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self._invoke_llm_with_retry(messages)
        return response.content.strip()