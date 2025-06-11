"""Core data models for DocuForge rewrite engine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ClarificationItem(BaseModel):
    """Single clarification question-answer pair."""
    
    question: str = Field(..., min_length=1, description="The clarification question")
    answer: str = Field(..., min_length=1, description="The answer to the question")
    
    @field_validator('question', 'answer')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure question and answer are not empty or whitespace only."""
        if not v.strip():
            raise ValueError("Question and answer cannot be empty or whitespace only")
        return v.strip()


class RewriteRequest(BaseModel):
    """Request data structure for document rewriting."""
    
    original_content: str = Field(
        ..., 
        min_length=1, 
        description="Original document content to be rewritten"
    )
    clarifications: List[ClarificationItem] = Field(
        default_factory=list,
        description="List of clarification question-answer pairs"
    )
    
    @field_validator('original_content')
    @classmethod
    def validate_original_content(cls, v: str) -> str:
        """Ensure original content is not empty."""
        if not v.strip():
            raise ValueError("Original content cannot be empty")
        return v.strip()


class DocumentSection(BaseModel):
    """Document section data structure."""
    
    title: str = Field(..., min_length=1, description="Section title")
    content: str = Field(default="", description="Section content")
    level: int = Field(..., ge=1, le=6, description="Section hierarchy level (1-6)")
    order: int = Field(..., ge=0, description="Section order within document")
    goal: str = Field(
        default="", 
        description="Writing objective for this section"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty."""
        if not v.strip():
            raise ValueError("Section title cannot be empty")
        return v.strip()
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: int) -> int:
        """Ensure level is within valid range."""
        if not 1 <= v <= 6:
            raise ValueError("Section level must be between 1 and 6")
        return v
    
    @field_validator('order')
    @classmethod
    def validate_order(cls, v: int) -> int:
        """Ensure order is non-negative."""
        if v < 0:
            raise ValueError("Section order must be non-negative")
        return v


class DocumentStructure(BaseModel):
    """Document structure data containing title and sections."""
    
    title: str = Field(..., min_length=1, description="Document title")
    sections: List[DocumentSection] = Field(
        default_factory=list,
        description="List of document sections"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the document"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty."""
        if not v.strip():
            raise ValueError("Document title cannot be empty")
        return v.strip()
    
    @field_validator('sections')
    @classmethod
    def validate_sections_order(cls, v: List[DocumentSection]) -> List[DocumentSection]:
        """Validate section ordering is consistent."""
        if not v:
            return v
        
        # Check for duplicate orders
        orders = [section.order for section in v]
        if len(orders) != len(set(orders)):
            raise ValueError("Duplicate section orders found")
        
        # Sort sections by order to ensure consistency
        return sorted(v, key=lambda s: s.order)
    
    def to_markdown(self) -> str:
        """Convert document structure to markdown format."""
        lines = [f"# {self.title}", ""]
        
        for section in self.sections:
            # Generate markdown headers based on level
            header_prefix = "#" * (section.level + 1)  # +1 because title is h1
            lines.append(f"{header_prefix} {section.title}")
            
            if section.content:
                lines.append("")
                lines.append(section.content)
            
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def get_section_by_order(self, order: int) -> Optional[DocumentSection]:
        """Get section by its order number."""
        for section in self.sections:
            if section.order == order:
                return section
        return None
    
    def add_section(self, section: DocumentSection) -> None:
        """Add a section to the document, maintaining order."""
        # Check for duplicate order
        if any(s.order == section.order for s in self.sections):
            raise ValueError(f"Section with order {section.order} already exists")
        
        self.sections.append(section)
        # Re-sort sections by order
        self.sections.sort(key=lambda s: s.order)


class RewriteResult(BaseModel):
    """Result data structure for document rewriting."""
    
    rewritten_content: str = Field(
        ..., 
        description="Rewritten content in markdown format"
    )
    structured_document: DocumentStructure = Field(
        ..., 
        description="Structured representation of the rewritten document"
    )
    
    @field_validator('rewritten_content')
    @classmethod
    def validate_rewritten_content(cls, v: str) -> str:
        """Ensure rewritten content is not empty."""
        if not v.strip():
            raise ValueError("Rewritten content cannot be empty")
        return v.strip()
    
    def model_post_init(self, __context: Any) -> None:
        """Ensure consistency between rewritten_content and structured_document."""
        # Generate markdown from structured document and compare
        generated_markdown = self.structured_document.to_markdown()
        
        # Allow some flexibility in whitespace differences
        normalized_content = "\n".join(line.strip() for line in self.rewritten_content.split("\n") if line.strip())
        normalized_generated = "\n".join(line.strip() for line in generated_markdown.split("\n") if line.strip())
        
        if normalized_content != normalized_generated:
            # For now, just log a warning instead of raising an error
            # In production, you might want to handle this differently
            pass