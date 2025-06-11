"""Unit tests for core data models."""

import pytest
from pydantic import ValidationError

from docuforge.models import (
    ClarificationItem,
    DocumentSection,
    DocumentStructure,
    RewriteRequest,
    RewriteResult,
)


class TestClarificationItem:
    """Test ClarificationItem model."""
    
    def test_valid_creation(self):
        """Test creating a valid ClarificationItem."""
        item = ClarificationItem(
            question="What is the main purpose?",
            answer="To provide AI-powered document rewriting"
        )
        assert item.question == "What is the main purpose?"
        assert item.answer == "To provide AI-powered document rewriting"
    
    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed from question and answer."""
        item = ClarificationItem(
            question="  What is the purpose?  ",
            answer="  To rewrite documents  "
        )
        assert item.question == "What is the purpose?"
        assert item.answer == "To rewrite documents"
    
    def test_empty_question_validation(self):
        """Test that empty question raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ClarificationItem(question="", answer="Valid answer")
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("question" in str(error) for error in errors)
    
    def test_empty_answer_validation(self):
        """Test that empty answer raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ClarificationItem(question="Valid question?", answer="")
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("answer" in str(error) for error in errors)
    
    def test_whitespace_only_validation(self):
        """Test that whitespace-only strings raise ValidationError."""
        with pytest.raises(ValidationError):
            ClarificationItem(question="   ", answer="Valid answer")
        
        with pytest.raises(ValidationError):
            ClarificationItem(question="Valid question?", answer="   ")
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original = ClarificationItem(
            question="How does it work?",
            answer="Using AI models"
        )
        
        # Test serialization
        json_data = original.model_dump()
        expected = {
            "question": "How does it work?",
            "answer": "Using AI models"
        }
        assert json_data == expected
        
        # Test deserialization
        recreated = ClarificationItem.model_validate(json_data)
        assert recreated == original


class TestRewriteRequest:
    """Test RewriteRequest model."""
    
    def test_valid_creation(self):
        """Test creating a valid RewriteRequest."""
        clarifications = [
            ClarificationItem(question="Q1?", answer="A1"),
            ClarificationItem(question="Q2?", answer="A2")
        ]
        
        request = RewriteRequest(
            original_content="Original document content",
            clarifications=clarifications
        )
        
        assert request.original_content == "Original document content"
        assert len(request.clarifications) == 2
        assert request.clarifications[0].question == "Q1?"
    
    def test_empty_clarifications_default(self):
        """Test that clarifications defaults to empty list."""
        request = RewriteRequest(original_content="Content")
        assert request.clarifications == []
    
    def test_content_whitespace_trimming(self):
        """Test that original content whitespace is trimmed."""
        request = RewriteRequest(original_content="  Content  ")
        assert request.original_content == "Content"
    
    def test_empty_content_validation(self):
        """Test that empty content raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RewriteRequest(original_content="")
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("original_content" in str(error) for error in errors)
    
    def test_whitespace_only_content_validation(self):
        """Test that whitespace-only content raises ValidationError."""
        with pytest.raises(ValidationError):
            RewriteRequest(original_content="   ")
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        clarifications = [ClarificationItem(question="Q?", answer="A")]
        original = RewriteRequest(
            original_content="Content",
            clarifications=clarifications
        )
        
        json_data = original.model_dump()
        recreated = RewriteRequest.model_validate(json_data)
        assert recreated == original


class TestDocumentSection:
    """Test DocumentSection model."""
    
    def test_valid_creation(self):
        """Test creating a valid DocumentSection."""
        section = DocumentSection(
            title="Introduction",
            content="This is the introduction content",
            level=1,
            order=0,
            goal="Introduce the topic"
        )
        
        assert section.title == "Introduction"
        assert section.content == "This is the introduction content"
        assert section.level == 1
        assert section.order == 0
        assert section.goal == "Introduce the topic"
    
    def test_default_values(self):
        """Test default values for optional fields."""
        section = DocumentSection(
            title="Title",
            level=2,
            order=1
        )
        
        assert section.content == ""
        assert section.goal == ""
    
    def test_title_whitespace_trimming(self):
        """Test that title whitespace is trimmed."""
        section = DocumentSection(
            title="  Section Title  ",
            level=1,
            order=0
        )
        assert section.title == "Section Title"
    
    def test_empty_title_validation(self):
        """Test that empty title raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentSection(title="", level=1, order=0)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("title" in str(error) for error in errors)
    
    def test_whitespace_only_title_validation(self):
        """Test that whitespace-only title raises ValidationError."""
        for title in ["   ", "\t", "\n", " \t \n "]:
            with pytest.raises(ValidationError) as exc_info:
                DocumentSection(title=title, level=1, order=0)
            
            errors = exc_info.value.errors()
            assert len(errors) >= 1
            assert any("Section title cannot be empty" in str(error) for error in errors)
    
    def test_level_validation(self):
        """Test level validation (must be 1-6)."""
        # Valid levels
        for level in [1, 2, 3, 4, 5, 6]:
            section = DocumentSection(title="Title", level=level, order=0)
            assert section.level == level
        
        # Invalid levels
        for level in [0, 7, -1, 10]:
            with pytest.raises(ValidationError):
                DocumentSection(title="Title", level=level, order=0)
    
    def test_order_validation(self):
        """Test order validation (must be non-negative)."""
        # Valid orders
        for order in [0, 1, 10, 100]:
            section = DocumentSection(title="Title", level=1, order=order)
            assert section.order == order
        
        # Invalid orders
        for order in [-1, -10]:
            with pytest.raises(ValidationError):
                DocumentSection(title="Title", level=1, order=order)
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original = DocumentSection(
            title="Test Section",
            content="Test content",
            level=2,
            order=1,
            goal="Test goal"
        )
        
        json_data = original.model_dump()
        recreated = DocumentSection.model_validate(json_data)
        assert recreated == original


class TestDocumentStructure:
    """Test DocumentStructure model."""
    
    def test_valid_creation(self):
        """Test creating a valid DocumentStructure."""
        sections = [
            DocumentSection(title="Intro", level=1, order=0),
            DocumentSection(title="Body", level=1, order=1)
        ]
        
        doc = DocumentStructure(
            title="Test Document",
            sections=sections,
            metadata={"author": "Test Author"}
        )
        
        assert doc.title == "Test Document"
        assert len(doc.sections) == 2
        assert doc.metadata["author"] == "Test Author"
    
    def test_default_values(self):
        """Test default values for optional fields."""
        doc = DocumentStructure(title="Title")
        assert doc.sections == []
        assert doc.metadata == {}
    
    def test_title_whitespace_trimming(self):
        """Test that title whitespace is trimmed."""
        doc = DocumentStructure(title="  Document Title  ")
        assert doc.title == "Document Title"
    
    def test_empty_title_validation(self):
        """Test that empty title raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentStructure(title="")
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("title" in str(error) for error in errors)
    
    def test_whitespace_only_title_validation(self):
        """Test that whitespace-only title raises ValidationError."""
        for title in ["   ", "\t", "\n", " \t \n "]:
            with pytest.raises(ValidationError) as exc_info:
                DocumentStructure(title=title)
            
            errors = exc_info.value.errors()
            assert len(errors) >= 1
            assert any("Document title cannot be empty" in str(error) for error in errors)
    
    def test_sections_ordering(self):
        """Test that sections are automatically sorted by order."""
        sections = [
            DocumentSection(title="Third", level=1, order=2),
            DocumentSection(title="First", level=1, order=0),
            DocumentSection(title="Second", level=1, order=1)
        ]
        
        doc = DocumentStructure(title="Test", sections=sections)
        
        # Sections should be sorted by order
        assert doc.sections[0].title == "First"
        assert doc.sections[1].title == "Second"
        assert doc.sections[2].title == "Third"
    
    def test_duplicate_order_validation(self):
        """Test that duplicate section orders raise ValidationError."""
        sections = [
            DocumentSection(title="First", level=1, order=0),
            DocumentSection(title="Duplicate", level=1, order=0)
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            DocumentStructure(title="Test", sections=sections)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("order" in str(error) for error in errors)
    
    def test_to_markdown(self):
        """Test markdown conversion."""
        sections = [
            DocumentSection(
                title="Introduction", 
                content="This is intro content",
                level=1, 
                order=0
            ),
            DocumentSection(
                title="Details",
                content="Detailed information here",
                level=2,
                order=1
            ),
            DocumentSection(
                title="Conclusion",
                content="",  # Empty content
                level=1,
                order=2
            )
        ]
        
        doc = DocumentStructure(title="Test Document", sections=sections)
        markdown = doc.to_markdown()
        
        expected_lines = [
            "# Test Document",
            "",
            "## Introduction",
            "",
            "This is intro content",
            "",
            "### Details",
            "",
            "Detailed information here",
            "",
            "## Conclusion"
        ]
        
        assert markdown == "\n".join(expected_lines)
    
    def test_get_section_by_order(self):
        """Test getting section by order."""
        sections = [
            DocumentSection(title="First", level=1, order=0),
            DocumentSection(title="Second", level=1, order=1)
        ]
        
        doc = DocumentStructure(title="Test", sections=sections)
        
        section = doc.get_section_by_order(1)
        assert section is not None
        assert section.title == "Second"
        
        section = doc.get_section_by_order(99)
        assert section is None
    
    def test_add_section(self):
        """Test adding sections."""
        doc = DocumentStructure(title="Test")
        
        section1 = DocumentSection(title="First", level=1, order=0)
        section2 = DocumentSection(title="Second", level=1, order=1)
        
        doc.add_section(section2)  # Add in reverse order
        doc.add_section(section1)
        
        # Should be sorted by order
        assert doc.sections[0].title == "First"
        assert doc.sections[1].title == "Second"
        
        # Test duplicate order
        duplicate = DocumentSection(title="Duplicate", level=1, order=0)
        with pytest.raises(ValueError, match="already exists"):
            doc.add_section(duplicate)
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        sections = [DocumentSection(title="Test", level=1, order=0)]
        original = DocumentStructure(
            title="Test Doc",
            sections=sections,
            metadata={"version": "1.0"}
        )
        
        json_data = original.model_dump()
        recreated = DocumentStructure.model_validate(json_data)
        assert recreated == original


class TestRewriteResult:
    """Test RewriteResult model."""
    
    def test_valid_creation(self):
        """Test creating a valid RewriteResult."""
        doc = DocumentStructure(
            title="Test Document",
            sections=[DocumentSection(title="Section", level=1, order=0)]
        )
        
        result = RewriteResult(
            rewritten_content="# Test Document\n\n## Section",
            structured_document=doc
        )
        
        assert "Test Document" in result.rewritten_content
        assert result.structured_document.title == "Test Document"
    
    def test_content_whitespace_trimming(self):
        """Test that rewritten content whitespace is trimmed."""
        doc = DocumentStructure(title="Test")
        
        result = RewriteResult(
            rewritten_content="  Content  ",
            structured_document=doc
        )
        
        assert result.rewritten_content == "Content"
    
    def test_empty_content_validation(self):
        """Test that empty rewritten content raises ValidationError."""
        doc = DocumentStructure(title="Test")
        
        with pytest.raises(ValidationError) as exc_info:
            RewriteResult(rewritten_content="", structured_document=doc)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("rewritten_content" in str(error) for error in errors)
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        doc = DocumentStructure(title="Test")
        original = RewriteResult(
            rewritten_content="Content",
            structured_document=doc
        )
        
        json_data = original.model_dump()
        recreated = RewriteResult.model_validate(json_data)
        assert recreated == original


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        item = ClarificationItem(
            question="中文问题？",
            answer="中文回答，包含特殊字符：@#$%^&*()"
        )
        
        assert item.question == "中文问题？"
        assert item.answer == "中文回答，包含特殊字符：@#$%^&*()"
        
        # Test serialization
        json_data = item.model_dump()
        recreated = ClarificationItem.model_validate(json_data)
        assert recreated == item
    
    def test_very_long_content(self):
        """Test handling of very long content."""
        long_content = "A" * 10000  # 10k characters
        
        request = RewriteRequest(original_content=long_content)
        assert len(request.original_content) == 10000
        
        # Test serialization
        json_data = request.model_dump()
        recreated = RewriteRequest.model_validate(json_data)
        assert recreated == request
    
    def test_special_characters_in_titles(self):
        """Test handling of special characters in titles."""
        section = DocumentSection(
            title="Title with Special Characters: @#$%^&*()",
            level=1,
            order=0
        )
        
        assert section.title == "Title with Special Characters: @#$%^&*()"
        
        doc = DocumentStructure(
            title="Document with 特殊字符 and Émojis 🚀",
            sections=[section]
        )
        
        markdown = doc.to_markdown()
        assert "Document with 特殊字符 and Émojis 🚀" in markdown
        assert "Title with Special Characters: @#$%^&*()" in markdown