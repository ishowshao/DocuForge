"""Unit tests for core components."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.language_models import BaseLLM
from langchain_core.messages import BaseMessage

from docuforge.components import (
    ComponentBase,
    ContextBuilder,
    OutlineGenerator,
    ContentFiller,
    Reviser,
    RevisionIssue,
    RevisionReport,
)
from docuforge.models import (
    ClarificationItem,
    DocumentSection,
    DocumentStructure,
    RewriteRequest,
)
from docuforge.callbacks import DefaultCallbackHandler


class MockLLM:
    """Mock LLM for testing (not inheriting from BaseLLM to avoid Pydantic issues)."""
    
    def __init__(self, response_content: str = "mock response"):
        self.response_content = response_content
    
    def invoke(self, messages, **kwargs):
        mock_response = Mock()
        mock_response.content = self.response_content
        return mock_response


class TestComponentBase:
    """Test ComponentBase class."""
    
    def test_initialization_with_defaults(self):
        """Test ComponentBase initialization with default parameters."""
        llm = MockLLM()
        
        # Since ComponentBase is not abstract, we can test its initialization
        # but it shouldn't be used directly in practice
        component = ComponentBase(llm)
        assert component.llm == llm
        assert component.callback_handler is not None
        assert component.prompt_manager is not None


class TestContextBuilder:
    """Test ContextBuilder component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.builder = ContextBuilder(self.llm)
    
    def test_build_context_with_no_clarifications(self):
        """Test building context with no clarifications."""
        request = RewriteRequest(
            original_content="This is the original document content."
        )
        
        context = self.builder.build_context(request)
        
        expected = """=== 原始文档内容 ===
This is the original document content."""
        
        assert context == expected
    
    def test_build_context_with_clarifications(self):
        """Test building context with clarifications."""
        clarifications = [
            ClarificationItem(question="What is the goal?", answer="To improve efficiency"),
            ClarificationItem(question="Who is the target?", answer="Enterprise users")
        ]
        
        request = RewriteRequest(
            original_content="Original content here.",
            clarifications=clarifications
        )
        
        context = self.builder.build_context(request)
        
        assert "=== 原始文档内容 ===" in context
        assert "Original content here." in context
        assert "=== 澄清信息 ===" in context
        assert "问题 1: What is the goal?" in context
        assert "回答 1: To improve efficiency" in context
        assert "问题 2: Who is the target?" in context
        assert "回答 2: Enterprise users" in context
    
    def test_validate_context_valid(self):
        """Test context validation with valid context."""
        valid_context = "This is a valid context with sufficient content."
        assert self.builder.validate_context(valid_context) is True
    
    def test_validate_context_empty(self):
        """Test context validation with empty context."""
        assert self.builder.validate_context("") is False
        assert self.builder.validate_context("   ") is False
    
    def test_validate_context_too_short(self):
        """Test context validation with too short context."""
        assert self.builder.validate_context("short") is False
    
    def test_validate_context_too_long(self):
        """Test context validation with too long context."""
        very_long_context = "A" * 100001  # Over 100k characters
        assert self.builder.validate_context(very_long_context) is False


class TestOutlineGenerator:
    """Test OutlineGenerator component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock LLM response for valid JSON
        valid_json = '''
        {
            "title": "Test Document",
            "sections": [
                {
                    "title": "Introduction",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "Introduce the topic"
                },
                {
                    "title": "Details",
                    "content": "",
                    "level": 2,
                    "order": 1,
                    "goal": "Provide detailed information"
                }
            ],
            "metadata": {}
        }
        '''
        self.llm = MockLLM(response_content=valid_json)
        self.generator = OutlineGenerator(self.llm)
        self.mock_callback = Mock()
        self.generator.callback_handler = self.mock_callback
    
    def test_generate_outline_success(self):
        """Test successful outline generation."""
        context = "Test context for outline generation"
        
        outline = self.generator.generate_outline(context)
        
        assert isinstance(outline, DocumentStructure)
        assert outline.title == "Test Document"
        assert len(outline.sections) == 2
        
        # Check first section
        section1 = outline.sections[0]
        assert section1.title == "Introduction"
        assert section1.content == ""
        assert section1.level == 1
        assert section1.order == 0
        assert section1.goal == "Introduce the topic"
        
        # Check callbacks were called
        assert self.mock_callback.on_stage_progress.call_count >= 2
    
    def test_generate_outline_invalid_json(self):
        """Test outline generation with invalid JSON response."""
        invalid_llm = MockLLM(response_content="Invalid JSON response")
        generator = OutlineGenerator(invalid_llm)
        
        context = "Test context"
        
        with pytest.raises(ValueError, match="Failed to parse outline"):
            generator.generate_outline(context)


class TestContentFiller:
    """Test ContentFiller component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM(response_content="Generated section content")
        self.filler = ContentFiller(self.llm)
        self.mock_callback = Mock()
        self.filler.callback_handler = self.mock_callback
    
    def test_fill_content_single_section(self):
        """Test filling content for document with single section."""
        sections = [
            DocumentSection(
                title="Test Section",
                content="",
                level=1,
                order=0,
                goal="Test the section filling"
            )
        ]
        
        document = DocumentStructure(
            title="Test Document",
            sections=sections
        )
        
        context = "Original context for testing"
        
        filled_document = self.filler.fill_content(document, context)
        
        assert filled_document.title == "Test Document"
        assert len(filled_document.sections) == 1
        assert filled_document.sections[0].content == "Generated section content"
        
        # Check callbacks
        assert self.mock_callback.on_stage_progress.call_count >= 2
    
    def test_fill_content_multiple_sections(self):
        """Test filling content for document with multiple sections."""
        sections = [
            DocumentSection(title="Section 1", content="", level=1, order=0, goal="Goal 1"),
            DocumentSection(title="Section 2", content="", level=2, order=1, goal="Goal 2"),
            DocumentSection(title="Section 3", content="", level=1, order=2, goal="Goal 3"),
        ]
        
        document = DocumentStructure(title="Multi-Section Doc", sections=sections)
        context = "Test context"
        
        filled_document = self.filler.fill_content(document, context)
        
        # All sections should have content
        for section in filled_document.sections:
            assert section.content == "Generated section content"
        
        # Should call progress callback for each section
        progress_calls = [
            call for call in self.mock_callback.on_stage_progress.call_args_list
            if "Filling section" in str(call)
        ]
        assert len(progress_calls) == 3
    
    def test_build_previous_content_first_section(self):
        """Test building previous content for first section."""
        document = DocumentStructure(title="Test", sections=[])
        
        previous_content = self.filler._build_previous_content(document, 0)
        
        assert previous_content == ""
    
    def test_build_previous_content_later_section(self):
        """Test building previous content for later sections."""
        sections = [
            DocumentSection(title="Section 1", content="Content 1", level=1, order=0, goal=""),
            DocumentSection(title="Section 2", content="Content 2", level=2, order=1, goal=""),
            DocumentSection(title="Section 3", content="", level=1, order=2, goal=""),
        ]
        
        document = DocumentStructure(title="Test Doc", sections=sections)
        
        previous_content = self.filler._build_previous_content(document, 2)
        
        expected_lines = [
            "# Test Doc",
            "",
            "## Section 1",
            "",
            "Content 1",
            "",
            "### Section 2",
            "",
            "Content 2"
        ]
        expected = "\n".join(expected_lines)
        
        assert previous_content == expected


class TestReviser:
    """Test Reviser component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock review response (no issues)
        no_issues_json = '''
        {
            "issues": [],
            "overall_quality": "Good quality document"
        }
        '''
        
        self.llm = MockLLM(response_content=no_issues_json)
        self.reviser = Reviser(self.llm, max_revision_rounds=2)
        self.mock_callback = Mock()
        self.reviser.callback_handler = self.mock_callback
    
    def test_review_and_revise_no_issues(self):
        """Test review and revision when no issues are found."""
        sections = [
            DocumentSection(
                title="Test Section",
                content="Good content here",
                level=1,
                order=0,
                goal="Test goal"
            )
        ]
        
        document = DocumentStructure(title="Test Doc", sections=sections)
        context = "Original context"
        
        revised_document = self.reviser.review_and_revise(document, context)
        
        # Document should be unchanged
        assert revised_document.title == document.title
        assert len(revised_document.sections) == 1
        assert revised_document.sections[0].content == "Good content here"
        
        # Should call progress callbacks
        progress_calls = self.mock_callback.on_stage_progress.call_args_list
        assert len(progress_calls) >= 2
    
    def test_review_and_revise_with_issues(self):
        """Test review and revision when issues are found."""
        # Mock review response with issues
        issues_json = '''
        {
            "issues": [
                {
                    "section_order": 0,
                    "issue_type": "inconsistency",
                    "description": "Content is inconsistent",
                    "suggestion": "Fix the inconsistency"
                }
            ],
            "overall_quality": "Needs improvement"
        }
        '''
        
        # Create mock LLM that returns issues first, then revision content
        mock_responses = [issues_json, "Fixed content"]
        mock_llm = MockLLM()
        
        # Override invoke to cycle through responses
        def mock_invoke(messages, **kwargs):
            if mock_llm.call_count < len(mock_responses):
                content = mock_responses[mock_llm.call_count]
                mock_llm.call_count += 1
            else:
                content = "Default response"
            
            response = Mock()
            response.content = content
            return response
        
        mock_llm.invoke = mock_invoke
        mock_llm.call_count = 0
        
        reviser = Reviser(mock_llm, max_revision_rounds=2)
        reviser.callback_handler = Mock()
        
        sections = [
            DocumentSection(
                title="Test Section",
                content="Problematic content",
                level=1,
                order=0,
                goal="Test goal"
            )
        ]
        
        document = DocumentStructure(title="Test Doc", sections=sections)
        context = "Original context"
        
        revised_document = reviser.review_and_revise(document, context)
        
        # Content should be revised
        assert revised_document.sections[0].content == "Fixed content"
        
        # Should have called LLM twice (review + revision)
        assert mock_llm.call_count == 2
    
    def test_revision_issue_model(self):
        """Test RevisionIssue model."""
        issue = RevisionIssue(
            section_order=1,
            issue_type="missing_info",
            description="Missing important details",
            suggestion="Add more details about the process"
        )
        
        assert issue.section_order == 1
        assert issue.issue_type == "missing_info"
        assert issue.description == "Missing important details"
        assert issue.suggestion == "Add more details about the process"
    
    def test_revision_report_model(self):
        """Test RevisionReport model."""
        issues = [
            RevisionIssue(
                section_order=0,
                issue_type="inconsistency",
                description="Test issue",
                suggestion="Fix it"
            )
        ]
        
        report = RevisionReport(
            issues=issues,
            overall_quality="Needs work"
        )
        
        assert report.has_issues is True
        assert len(report.issues) == 1
        assert report.overall_quality == "Needs work"
        
        # Test empty report
        empty_report = RevisionReport(issues=[], overall_quality="Good")
        assert empty_report.has_issues is False
    
    def test_max_revision_rounds(self):
        """Test that revision rounds are limited."""
        # Mock LLM that always returns issues
        issues_json = '''
        {
            "issues": [
                {
                    "section_order": 0,
                    "issue_type": "quality_issue",
                    "description": "Always has issues",
                    "suggestion": "Never fixed"
                }
            ],
            "overall_quality": "Poor"
        }
        '''
        
        mock_llm = Mock(spec=BaseLLM)
        # Always return issues and revised content
        mock_llm.invoke.return_value = Mock(content=issues_json)
        
        reviser = Reviser(mock_llm, max_revision_rounds=2)
        reviser.callback_handler = Mock()
        
        sections = [DocumentSection(title="Test", content="Content", level=1, order=0, goal="")]
        document = DocumentStructure(title="Test", sections=sections)
        
        # Should stop after max rounds even if issues persist
        revised_document = reviser.review_and_revise(document, "context")
        
        # Should have made multiple LLM calls but stopped at max rounds
        # Each round: 1 review call + 1 revision call = 2 calls per round
        # Plus final review call = 2*2 + 1 = 5 calls maximum
        assert mock_llm.invoke.call_count <= 5


class TestIntegration:
    """Integration tests for components working together."""
    
    def test_component_chain_basic_flow(self):
        """Test basic flow of context -> outline -> content -> review."""
        # Setup mocks
        outline_json = '''
        {
            "title": "Integration Test Doc",
            "sections": [
                {
                    "title": "Overview",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "Provide an overview"
                }
            ],
            "metadata": {}
        }
        '''
        
        review_json = '''
        {
            "issues": [],
            "overall_quality": "Excellent"
        }
        '''
        
        # Create components
        context_builder = ContextBuilder(MockLLM())
        outline_generator = OutlineGenerator(MockLLM(outline_json))
        content_filler = ContentFiller(MockLLM("Detailed overview content"))
        reviser = Reviser(MockLLM(review_json))
        
        # Test data
        request = RewriteRequest(
            original_content="Original document about integration testing",
            clarifications=[
                ClarificationItem(question="What to test?", answer="Component integration")
            ]
        )
        
        # Step 1: Build context
        context = context_builder.build_context(request)
        assert "Original document about integration testing" in context
        assert "What to test?" in context
        
        # Step 2: Generate outline
        outline = outline_generator.generate_outline(context)
        assert outline.title == "Integration Test Doc"
        assert len(outline.sections) == 1
        assert outline.sections[0].content == ""  # Empty initially
        
        # Step 3: Fill content
        filled_document = content_filler.fill_content(outline, context)
        assert filled_document.sections[0].content == "Detailed overview content"
        
        # Step 4: Review and revise
        final_document = reviser.review_and_revise(filled_document, context)
        assert final_document.sections[0].content == "Detailed overview content"  # No changes needed
        
        # Verify final document structure
        assert final_document.title == "Integration Test Doc"
        assert len(final_document.sections) == 1
    
    def test_error_propagation(self):
        """Test that errors are properly propagated through components."""
        # Use an LLM that returns invalid JSON
        invalid_llm = MockLLM("Invalid JSON response")
        outline_generator = OutlineGenerator(invalid_llm)
        
        with pytest.raises(ValueError, match="Failed to parse outline"):
            outline_generator.generate_outline("test context")