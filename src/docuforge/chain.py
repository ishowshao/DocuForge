"""LangGraph-based rewrite chain implementation."""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.language_models import BaseLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph

from .callbacks import DefaultCallbackHandler, ProgressCallbackHandler
from .components import (
    ContentFiller,
    ContextBuilder,
    OutlineGenerator,
    Reviser,
    RevisionReport,
)
from .models import DocumentStructure, RewriteRequest, RewriteResult


class RewriteState(TypedDict):
    """State object for the rewrite workflow."""
    
    # Input data
    request: RewriteRequest
    context: str
    
    # Intermediate results
    outline: Optional[DocumentStructure]
    filled_document: Optional[DocumentStructure]
    revision_report: Optional[RevisionReport]
    
    # Control flow
    revision_round: int
    max_revision_rounds: int
    
    # Output
    final_document: Optional[DocumentStructure]
    error: Optional[str]


class RewriteChain:
    """Main rewrite chain orchestrator using LangGraph."""
    
    def __init__(
        self,
        llm: BaseLLM,
        callback_handler: Optional[ProgressCallbackHandler] = None,
        max_revision_rounds: int = 3
    ):
        """Initialize the rewrite chain.
        
        Args:
            llm: Language model to use for all components
            callback_handler: Progress callback handler
            max_revision_rounds: Maximum number of revision rounds
        """
        self.llm = llm
        self.callback_handler = callback_handler or DefaultCallbackHandler()
        self.max_revision_rounds = max_revision_rounds
        
        # Initialize components
        self.context_builder = ContextBuilder(llm, callback_handler)
        self.outline_generator = OutlineGenerator(llm, callback_handler)
        self.content_filler = ContentFiller(llm, callback_handler)
        self.reviser = Reviser(llm, callback_handler, max_revision_rounds=max_revision_rounds)
        
        # Build the workflow graph
        self.graph = self._build_graph()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def invoke(self, request: RewriteRequest) -> RewriteResult:
        """Execute the rewrite workflow.
        
        Args:
            request: Rewrite request containing original content and clarifications
            
        Returns:
            RewriteResult with rewritten content and structured document
            
        Raises:
            ValueError: If the workflow fails
        """
        try:
            self.callback_handler.on_stage_start("rewrite_workflow")
            
            # Initialize state
            initial_state: RewriteState = {
                "request": request,
                "context": "",
                "outline": None,
                "filled_document": None,
                "revision_report": None,
                "revision_round": 0,
                "max_revision_rounds": self.max_revision_rounds,
                "final_document": None,
                "error": None
            }
            
            # Execute the workflow
            final_state = self.graph.invoke(initial_state)
            
            if final_state["error"]:
                raise ValueError(f"Workflow failed: {final_state['error']}")
            
            if not final_state["final_document"]:
                raise ValueError("Workflow completed but no final document produced")
            
            # Create result
            final_document = final_state["final_document"]
            rewritten_content = final_document.to_markdown()
            
            result = RewriteResult(
                rewritten_content=rewritten_content,
                structured_document=final_document
            )
            
            self.callback_handler.on_stage_end("rewrite_workflow")
            return result
            
        except Exception as e:
            self.callback_handler.on_stage_progress(
                "rewrite_workflow",
                f"Workflow failed: {str(e)}"
            )
            self.callback_handler.on_stage_end("rewrite_workflow")
            raise
    
    def _build_graph(self) -> CompiledGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(RewriteState)
        
        # Add nodes
        graph.add_node("build_context", self._build_context_node)
        graph.add_node("generate_outline", self._generate_outline_node)
        graph.add_node("fill_content", self._fill_content_node)
        graph.add_node("review_document", self._review_document_node)
        graph.add_node("apply_revisions", self._apply_revisions_node)
        graph.add_node("finalize", self._finalize_node)
        
        # Define edges
        graph.add_edge(START, "build_context")
        graph.add_edge("build_context", "generate_outline")
        graph.add_edge("generate_outline", "fill_content")
        graph.add_edge("fill_content", "review_document")
        
        # Conditional edge for revision loop
        graph.add_conditional_edges(
            "review_document",
            self._should_revise,
            {
                "revise": "apply_revisions",
                "finalize": "finalize"
            }
        )
        
        graph.add_edge("apply_revisions", "review_document")
        graph.add_edge("finalize", END)
        
        return graph.compile()
    
    def _build_context_node(self, state: RewriteState) -> Dict[str, Any]:
        """Build unified context from request."""
        try:
            self.callback_handler.on_stage_start("context_building")
            
            context = self.context_builder.build_context(state["request"])
            
            if not self.context_builder.validate_context(context):
                return {
                    **state,
                    "error": "Invalid context: context is empty or too short"
                }
            
            self.callback_handler.on_stage_end("context_building")
            
            return {
                **state,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Context building failed: {e}")
            return {
                **state,
                "error": f"Context building failed: {str(e)}"
            }
    
    def _generate_outline_node(self, state: RewriteState) -> Dict[str, Any]:
        """Generate document outline."""
        try:
            self.callback_handler.on_stage_start("outline_generation")
            
            outline = self.outline_generator.generate_outline(state["context"])
            
            self.callback_handler.on_stage_end("outline_generation")
            
            return {
                **state,
                "outline": outline
            }
            
        except Exception as e:
            self.logger.error(f"Outline generation failed: {e}")
            return {
                **state,
                "error": f"Outline generation failed: {str(e)}"
            }
    
    def _fill_content_node(self, state: RewriteState) -> Dict[str, Any]:
        """Fill content for all sections."""
        try:
            self.callback_handler.on_stage_start("content_filling")
            
            if not state["outline"]:
                return {
                    **state,
                    "error": "No outline available for content filling"
                }
            
            filled_document = self.content_filler.fill_content(
                state["outline"], state["context"]
            )
            
            self.callback_handler.on_stage_end("content_filling")
            
            return {
                **state,
                "filled_document": filled_document
            }
            
        except Exception as e:
            self.logger.error(f"Content filling failed: {e}")
            return {
                **state,
                "error": f"Content filling failed: {str(e)}"
            }
    
    def _review_document_node(self, state: RewriteState) -> Dict[str, Any]:
        """Review the document for issues."""
        try:
            self.callback_handler.on_stage_start("review_revision")
            
            if not state["filled_document"]:
                return {
                    **state,
                    "error": "No document available for review"
                }
            
            # Use the reviser's review method
            report = self.reviser._review_document(
                state["filled_document"], state["context"]
            )
            
            self.callback_handler.on_stage_progress(
                "review_revision",
                f"Review completed: {len(report.issues)} issues found"
            )
            
            return {
                **state,
                "revision_report": report
            }
            
        except Exception as e:
            self.logger.error(f"Document review failed: {e}")
            return {
                **state,
                "error": f"Document review failed: {str(e)}"
            }
    
    def _apply_revisions_node(self, state: RewriteState) -> Dict[str, Any]:
        """Apply revisions to the document."""
        try:
            if not state["filled_document"] or not state["revision_report"]:
                return {
                    **state,
                    "error": "Missing document or revision report for applying revisions"
                }
            
            self.callback_handler.on_stage_progress(
                "review_revision",
                f"Applying {len(state['revision_report'].issues)} revisions"
            )
            
            # Apply revisions
            revised_document = self.reviser._apply_revisions(
                state["filled_document"],
                state["revision_report"],
                state["context"]
            )
            
            return {
                **state,
                "filled_document": revised_document,
                "revision_round": state["revision_round"] + 1
            }
            
        except Exception as e:
            self.logger.error(f"Applying revisions failed: {e}")
            return {
                **state,
                "error": f"Applying revisions failed: {str(e)}"
            }
    
    def _finalize_node(self, state: RewriteState) -> Dict[str, Any]:
        """Finalize the document."""
        self.callback_handler.on_stage_end("review_revision")
        
        return {
            **state,
            "final_document": state["filled_document"]
        }
    
    def _should_revise(self, state: RewriteState) -> str:
        """Determine whether to revise or finalize the document."""
        if state["error"]:
            return "finalize"
        
        if not state["revision_report"]:
            return "finalize"
        
        # Check if we have issues and haven't exceeded max rounds
        if (state["revision_report"].has_issues and 
            state["revision_round"] < state["max_revision_rounds"]):
            return "revise"
        
        return "finalize"
    
    def get_workflow_visualization(self) -> str:
        """Get a text representation of the workflow."""
        return """
RewriteChain Workflow:
===================

START
  ↓
build_context (ContextBuilder)
  ↓
generate_outline (OutlineGenerator)
  ↓
fill_content (ContentFiller)
  ↓
review_document (Reviser)
  ↓
[Conditional Decision]
  ├─ has_issues & revision_round < max → apply_revisions → review_document
  └─ no_issues | revision_round >= max → finalize
                                            ↓
                                           END

Components:
- ContextBuilder: Aggregates original content + clarifications
- OutlineGenerator: Creates DocumentStructure with goals but empty content
- ContentFiller: Fills content section by section with rolling context
- Reviser: Reviews document and applies targeted fixes
        """


# Utility function for easy chain creation
def create_rewrite_chain(
    llm: BaseLLM,
    callback_handler: Optional[ProgressCallbackHandler] = None,
    max_revision_rounds: int = 3
) -> RewriteChain:
    """Create a rewrite chain with the given parameters.
    
    Args:
        llm: Language model to use
        callback_handler: Optional progress callback handler
        max_revision_rounds: Maximum number of revision rounds
        
    Returns:
        Configured RewriteChain instance
    """
    return RewriteChain(
        llm=llm,
        callback_handler=callback_handler,
        max_revision_rounds=max_revision_rounds
    )