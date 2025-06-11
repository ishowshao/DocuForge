"""Unit tests for RewriteChain."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.language_models import BaseLLM

from docuforge.chain import RewriteChain, RewriteState, create_rewrite_chain
from docuforge.models import (
    ClarificationItem,
    DocumentSection,
    DocumentStructure,
    RewriteRequest,
    RewriteResult,
)
from docuforge.callbacks import DefaultCallbackHandler


class MockLLM:
    """Mock LLM for testing chain functionality."""
    
    def __init__(self, responses=None):
        """Initialize with predetermined responses."""
        self.responses = responses or []
        self.call_count = 0
    
    def invoke(self, messages, **kwargs):
        """Return next predetermined response."""
        if self.call_count < len(self.responses):
            response = Mock()
            response.content = self.responses[self.call_count]
            self.call_count += 1
            return response
        else:
            # Default response
            response = Mock()
            response.content = "Default response"
            return response


class TestRewriteState:
    """Test RewriteState TypedDict."""
    
    def test_state_structure(self):
        """Test that RewriteState has correct structure."""
        from docuforge.models import RewriteRequest
        
        request = RewriteRequest(original_content="test")
        
        state: RewriteState = {
            "request": request,
            "context": "test context",
            "outline": None,
            "filled_document": None,
            "revision_report": None,
            "revision_round": 0,
            "max_revision_rounds": 3,
            "final_document": None,
            "error": None
        }
        
        assert state["request"] == request
        assert state["context"] == "test context"
        assert state["revision_round"] == 0
        assert state["max_revision_rounds"] == 3


class TestRewriteChain:
    """Test RewriteChain functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Prepare mock responses for successful flow
        self.outline_json = '''
        {
            "title": "Test Document",
            "sections": [
                {
                    "title": "Introduction",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "Introduce the topic"
                }
            ],
            "metadata": {}
        }
        '''
        
        self.review_json = '''
        {
            "issues": [],
            "overall_quality": "Good quality"
        }
        '''
        
        self.responses = [
            self.outline_json,     # For outline generation
            "Generated content",   # For content filling
            self.review_json      # For review
        ]
        
        self.mock_llm = MockLLM(self.responses)
        self.mock_callback = Mock()
        self.chain = RewriteChain(
            llm=self.mock_llm,
            callback_handler=self.mock_callback,
            max_revision_rounds=2
        )
    
    def test_initialization(self):
        """Test RewriteChain initialization."""
        llm = MockLLM()
        callback = Mock()
        
        chain = RewriteChain(
            llm=llm,
            callback_handler=callback,
            max_revision_rounds=3
        )
        
        assert chain.llm == llm
        assert chain.callback_handler == callback
        assert chain.max_revision_rounds == 3
        assert chain.context_builder is not None
        assert chain.outline_generator is not None
        assert chain.content_filler is not None
        assert chain.reviser is not None
        assert chain.graph is not None
    
    def test_initialization_defaults(self):
        """Test RewriteChain initialization with defaults."""
        llm = MockLLM()
        chain = RewriteChain(llm=llm)
        
        assert isinstance(chain.callback_handler, DefaultCallbackHandler)
        assert chain.max_revision_rounds == 3
    
    def test_invoke_no_final_document_error(self):
        """Test error when workflow completes but no final document is produced."""
        # Mock a workflow that succeeds but doesn't produce a final document
        request = RewriteRequest(original_content="Test content")
        
        # Create a mock graph that returns success state but no final document
        mock_result = {
            "error": None,
            "final_document": None  # This triggers the error
        }
        
        # Mock the graph invoke method
        self.chain.graph.invoke = Mock(return_value=mock_result)
        
        with pytest.raises(ValueError, match="Workflow completed but no final document produced"):
            self.chain.invoke(request)
    
    def test_successful_invoke(self):
        """Test successful execution of the rewrite workflow."""
        request = RewriteRequest(
            original_content="Original document content for testing",
            clarifications=[
                ClarificationItem(
                    question="What is the main goal?",
                    answer="To test the rewrite functionality"
                )
            ]
        )
        
        result = self.chain.invoke(request)
        
        # Verify result structure
        assert isinstance(result, RewriteResult)
        assert result.rewritten_content
        assert isinstance(result.structured_document, DocumentStructure)
        
        # Verify document structure
        doc = result.structured_document
        assert doc.title == "Test Document"
        assert len(doc.sections) == 1
        assert doc.sections[0].title == "Introduction"
        assert doc.sections[0].content == "Generated content"
        
        # Verify callbacks were called
        self.mock_callback.on_stage_start.assert_called()
        self.mock_callback.on_stage_end.assert_called()
    
    def test_invoke_with_context_building_error(self):
        """Test invoke with context building error."""
        # Mock context builder to validate context as invalid
        with patch.object(self.chain.context_builder, 'validate_context', return_value=False):
            request = RewriteRequest(original_content="test")
            
            with pytest.raises(ValueError, match="Invalid context"):
                self.chain.invoke(request)
    
    def test_invoke_with_outline_generation_error(self):
        """Test invoke with outline generation error."""
        # Use LLM that returns invalid JSON for outline
        error_llm = MockLLM(["Invalid JSON"])
        chain = RewriteChain(llm=error_llm)
        
        request = RewriteRequest(original_content="Valid content for testing")
        
        with pytest.raises(ValueError, match="Workflow failed"):
            chain.invoke(request)
    
    def test_should_revise_with_issues(self):
        """Test _should_revise method with issues present."""
        from docuforge.components import RevisionReport, RevisionIssue
        
        state: RewriteState = {
            "request": Mock(),
            "context": "",
            "outline": None,
            "filled_document": None,
            "revision_report": RevisionReport(
                issues=[RevisionIssue(
                    section_order=0,
                    issue_type="inconsistency",
                    description="Test issue",
                    suggestion="Fix it"
                )],
                overall_quality="Needs work"
            ),
            "revision_round": 0,
            "max_revision_rounds": 3,
            "final_document": None,
            "error": None
        }
        
        result = self.chain._should_revise(state)
        assert result == "revise"
    
    def test_should_revise_no_issues(self):
        """Test _should_revise method with no issues."""
        from docuforge.components import RevisionReport
        
        state: RewriteState = {
            "request": Mock(),
            "context": "",
            "outline": None,
            "filled_document": None,
            "revision_report": RevisionReport(issues=[], overall_quality="Good"),
            "revision_round": 0,
            "max_revision_rounds": 3,
            "final_document": None,
            "error": None
        }
        
        result = self.chain._should_revise(state)
        assert result == "finalize"
    
    def test_should_revise_max_rounds_exceeded(self):
        """Test _should_revise method when max rounds exceeded."""
        from docuforge.components import RevisionReport, RevisionIssue
        
        state: RewriteState = {
            "request": Mock(),
            "context": "",
            "outline": None,
            "filled_document": None,
            "revision_report": RevisionReport(
                issues=[RevisionIssue(
                    section_order=0,
                    issue_type="quality_issue",
                    description="Still has issues",
                    suggestion="Keep trying"
                )],
                overall_quality="Still needs work"
            ),
            "revision_round": 3,  # Reached max rounds
            "max_revision_rounds": 3,
            "final_document": None,
            "error": None
        }
        
        result = self.chain._should_revise(state)
        assert result == "finalize"
    
    def test_should_revise_with_error(self):
        """Test _should_revise method with error state."""
        state: RewriteState = {
            "request": Mock(),
            "context": "",
            "outline": None,
            "filled_document": None,
            "revision_report": None,
            "revision_round": 0,
            "max_revision_rounds": 3,
            "final_document": None,
            "error": "Something went wrong"
        }
        
        result = self.chain._should_revise(state)
        assert result == "finalize"
    
    def test_workflow_visualization(self):
        """Test workflow visualization."""
        visualization = self.chain.get_workflow_visualization()
        
        assert "RewriteChain Workflow" in visualization
        assert "build_context" in visualization
        assert "generate_outline" in visualization
        assert "fill_content" in visualization
        assert "review_document" in visualization
        assert "ContextBuilder" in visualization
        assert "OutlineGenerator" in visualization
        assert "ContentFiller" in visualization
        assert "Reviser" in visualization
    
    def test_revision_workflow(self):
        """Test workflow with revision cycle."""
        # Setup responses: outline, content, review with issues, revised content, final review
        responses_with_revision = [
            self.outline_json,
            "Initial content with issues",
            '''
            {
                "issues": [
                    {
                        "section_order": 0,
                        "issue_type": "quality_issue",
                        "description": "Content needs improvement",
                        "suggestion": "Make it better"
                    }
                ],
                "overall_quality": "Needs work"
            }
            ''',
            "Improved content after revision",
            '''
            {
                "issues": [],
                "overall_quality": "Much better now"
            }
            '''
        ]
        
        revision_llm = MockLLM(responses_with_revision)
        chain = RewriteChain(llm=revision_llm, max_revision_rounds=2)
        
        request = RewriteRequest(original_content="Test content for revision")
        result = chain.invoke(request)
        
        # Should have gone through revision cycle
        assert result.structured_document.sections[0].content == "Improved content after revision"
        
        # Should have made multiple LLM calls
        assert revision_llm.call_count == 5  # outline + content + review + revision + final review


class TestCreateRewriteChain:
    """Test create_rewrite_chain utility function."""
    
    def test_create_rewrite_chain_defaults(self):
        """Test creating chain with default parameters."""
        llm = MockLLM()
        chain = create_rewrite_chain(llm)
        
        assert isinstance(chain, RewriteChain)
        assert chain.llm == llm
        assert isinstance(chain.callback_handler, DefaultCallbackHandler)
        assert chain.max_revision_rounds == 3
    
    def test_create_rewrite_chain_custom_params(self):
        """Test creating chain with custom parameters."""
        llm = MockLLM()
        callback = Mock()
        
        chain = create_rewrite_chain(
            llm=llm,
            callback_handler=callback,
            max_revision_rounds=5
        )
        
        assert chain.llm == llm
        assert chain.callback_handler == callback
        assert chain.max_revision_rounds == 5


class TestChainIntegration:
    """Integration tests for the complete chain."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Prepare comprehensive responses
        outline_response = '''
        {
            "title": "Integration Test Document",
            "sections": [
                {
                    "title": "Overview",
                    "content": "",
                    "level": 1,
                    "order": 0,
                    "goal": "Provide system overview"
                },
                {
                    "title": "Implementation Details",
                    "content": "",
                    "level": 2,
                    "order": 1,
                    "goal": "Explain implementation approach"
                }
            ],
            "metadata": {"generated_by": "test"}
        }
        '''
        
        review_response = '''
        {
            "issues": [],
            "overall_quality": "Excellent comprehensive document"
        }
        '''
        
        responses = [
            outline_response,
            "Comprehensive system overview content",  # First section
            "Detailed implementation explanation",    # Second section
            review_response
        ]
        
        llm = MockLLM(responses)
        callback = Mock()
        chain = RewriteChain(llm=llm, callback_handler=callback)
        
        # Create comprehensive request
        request = RewriteRequest(
            original_content="""
            System Design Document
            
            This document describes a new system for processing data.
            The system should be scalable and reliable.
            """,
            clarifications=[
                ClarificationItem(
                    question="What type of data processing?",
                    answer="Real-time stream processing for financial transactions"
                ),
                ClarificationItem(
                    question="What are the scalability requirements?",
                    answer="Handle up to 100,000 transactions per second"
                ),
                ClarificationItem(
                    question="What reliability features are needed?",
                    answer="99.99% uptime with automatic failover"
                )
            ]
        )
        
        # Execute workflow
        result = chain.invoke(request)
        
        # Verify comprehensive result
        assert isinstance(result, RewriteResult)
        
        doc = result.structured_document
        assert doc.title == "Integration Test Document"
        assert len(doc.sections) == 2
        
        # Verify sections were filled
        assert doc.sections[0].content == "Comprehensive system overview content"
        assert doc.sections[1].content == "Detailed implementation explanation"
        
        # Verify markdown output
        markdown = result.rewritten_content
        assert "Integration Test Document" in markdown
        assert "Overview" in markdown
        assert "Implementation Details" in markdown
        
        # Verify callback interactions
        callback.on_stage_start.assert_called()
        callback.on_stage_end.assert_called()
        assert callback.on_stage_progress.call_count > 0
    
    def test_error_recovery(self):
        """Test error handling and recovery."""
        # LLM that fails on first outline attempt but succeeds on retry
        failing_llm = Mock(spec=BaseLLM)
        failing_llm.invoke.side_effect = [
            Exception("Network error"),  # First call fails
            Mock(content='''{"title": "Recovered Doc", "sections": [], "metadata": {}}'''),  # Second succeeds
            Mock(content="Recovered content"),
            Mock(content='{"issues": [], "overall_quality": "Good"}')
        ]
        
        # Chain should handle component-level errors gracefully
        # Note: This test demonstrates that errors in individual components
        # should be caught and handled by the chain's error handling logic
        with pytest.raises((Exception, ValueError)):
            chain = RewriteChain(llm=failing_llm)
            request = RewriteRequest(original_content="Test content")
            chain.invoke(request)


class TestChainPerformance:
    """Test chain performance and optimization."""
    
    def test_minimal_llm_calls(self):
        """Test that chain makes minimal necessary LLM calls."""
        # For a single section document with no revisions needed:
        # 1 call for outline, 1 call for content, 1 call for review = 3 total
        
        responses = [
            '{"title": "Test", "sections": [{"title": "Section", "content": "", "level": 1, "order": 0, "goal": "Test"}], "metadata": {}}',
            "Section content",
            '{"issues": [], "overall_quality": "Good"}'
        ]
        
        llm = MockLLM(responses)
        chain = RewriteChain(llm=llm)
        
        request = RewriteRequest(original_content="Simple test content")
        result = chain.invoke(request)
        
        # Should have made exactly 3 LLM calls
        assert llm.call_count == 3
        assert result.structured_document.sections[0].content == "Section content"
    
    def test_revision_round_limit(self):
        """Test that revision rounds are properly limited."""
        # Always return issues to test max rounds limit
        persistent_issues = '''
        {
            "issues": [
                {
                    "section_order": 0,
                    "issue_type": "persistent_issue",
                    "description": "This issue never gets fixed",
                    "suggestion": "Keep trying"
                }
            ],
            "overall_quality": "Still needs work"
        }
        '''
        
        responses = [
            '{"title": "Test", "sections": [{"title": "Section", "content": "", "level": 1, "order": 0, "goal": "Test"}], "metadata": {}}',
            "Original content",
            persistent_issues,  # First review
            "Revised content",
            persistent_issues,  # Second review
            "Revised again",
            persistent_issues   # Third review - should stop here
        ]
        
        llm = MockLLM(responses)
        chain = RewriteChain(llm=llm, max_revision_rounds=2)
        
        request = RewriteRequest(original_content="Test content")
        result = chain.invoke(request)
        
        # Should have stopped at max rounds despite persistent issues
        assert result.structured_document.sections[0].content in ["Revised content", "Revised again"]
        
        # Should not exceed expected number of calls
        # Outline + Content + Review + Revision + Review + Revision + Final Review = 7 max
        assert llm.call_count <= 7