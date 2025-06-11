"""Unit tests for callback system."""

import io
import logging
import sys
from unittest.mock import Mock, patch

import pytest

from docuforge.callbacks import (
    CLICallbackHandler,
    CompositeCallbackHandler,
    DefaultCallbackHandler,
    LoggingCallbackHandler,
    ProgressCallbackHandler,
)


class TestProgressCallbackHandler:
    """Test abstract ProgressCallbackHandler."""
    
    def test_is_abstract(self):
        """Test that ProgressCallbackHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProgressCallbackHandler()
    
    def test_abstract_methods_coverage(self):
        """Test coverage of abstract method implementations in concrete class."""
        # This tests the actual pass statements in abstract methods
        class TestHandler(ProgressCallbackHandler):
            def on_stage_start(self, stage_name: str, **kwargs):
                pass  # This covers line 14
            
            def on_stage_end(self, stage_name: str, **kwargs):
                pass  # This covers line 19
            
            def on_stage_progress(self, stage_name: str, message: str, **kwargs):
                pass  # This covers line 24
        
        handler = TestHandler()
        handler.on_stage_start("test")
        handler.on_stage_end("test")
        handler.on_stage_progress("test", "message")


class TestDefaultCallbackHandler:
    """Test DefaultCallbackHandler."""
    
    def test_instantiation(self):
        """Test that DefaultCallbackHandler can be instantiated."""
        handler = DefaultCallbackHandler()
        assert isinstance(handler, ProgressCallbackHandler)
    
    def test_all_methods_no_op(self):
        """Test that all methods are no-op and don't raise exceptions."""
        handler = DefaultCallbackHandler()
        
        # Should not raise any exceptions
        handler.on_stage_start("test_stage", extra_param="value")
        handler.on_stage_end("test_stage", extra_param="value")
        handler.on_stage_progress("test_stage", "test message", extra_param="value")


class TestCLICallbackHandler:
    """Test CLICallbackHandler."""
    
    def test_instantiation(self):
        """Test CLICallbackHandler instantiation."""
        handler = CLICallbackHandler()
        assert handler.verbose is True
        
        handler_quiet = CLICallbackHandler(verbose=False)
        assert handler_quiet.verbose is False
    
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_verbose_output(self, mock_stderr):
        """Test verbose output to stderr."""
        handler = CLICallbackHandler(verbose=True)
        
        handler.on_stage_start("test_stage")
        handler.on_stage_end("test_stage")
        handler.on_stage_progress("test_stage", "progress message")
        
        output = mock_stderr.getvalue()
        assert "INFO: Stage start: test_stage..." in output
        assert "INFO: Stage end: test_stage." in output
        assert "INFO: [test_stage] progress message" in output
    
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_quiet_output(self, mock_stderr):
        """Test that quiet mode produces no output."""
        handler = CLICallbackHandler(verbose=False)
        
        handler.on_stage_start("test_stage")
        handler.on_stage_end("test_stage")
        handler.on_stage_progress("test_stage", "progress message")
        
        output = mock_stderr.getvalue()
        assert output == ""
    
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_kwargs_ignored(self, mock_stderr):
        """Test that extra kwargs are ignored gracefully."""
        handler = CLICallbackHandler(verbose=True)
        
        handler.on_stage_start("test_stage", extra_param="value", another=123)
        handler.on_stage_end("test_stage", extra_param="value")
        handler.on_stage_progress("test_stage", "message", extra_param="value")
        
        output = mock_stderr.getvalue()
        assert "INFO: Stage start: test_stage..." in output
        assert "INFO: Stage end: test_stage." in output
        assert "INFO: [test_stage] message" in output


class TestLoggingCallbackHandler:
    """Test LoggingCallbackHandler."""
    
    def test_instantiation_default_logger(self):
        """Test instantiation with default logger."""
        handler = LoggingCallbackHandler()
        assert handler.logger is not None
        assert isinstance(handler.logger, logging.Logger)
    
    def test_instantiation_custom_logger(self):
        """Test instantiation with custom logger."""
        custom_logger = Mock(spec=logging.Logger)
        handler = LoggingCallbackHandler(logger=custom_logger)
        assert handler.logger is custom_logger
    
    def test_logging_calls(self):
        """Test that logging methods are called correctly."""
        mock_logger = Mock(spec=logging.Logger)
        handler = LoggingCallbackHandler(logger=mock_logger)
        
        handler.on_stage_start("test_stage", extra_param="value")
        handler.on_stage_end("test_stage", extra_param="value")
        handler.on_stage_progress("test_stage", "progress message", extra_param="value")
        
        # Check that info was called with correct messages
        assert mock_logger.info.call_count == 3
        
        calls = mock_logger.info.call_args_list
        assert calls[0].args[0] == "Stage start: test_stage"
        assert calls[0].kwargs == {"extra": {"extra_param": "value"}}
        
        assert calls[1].args[0] == "Stage end: test_stage"
        assert calls[1].kwargs == {"extra": {"extra_param": "value"}}
        
        assert calls[2].args[0] == "[test_stage] progress message"
        assert calls[2].kwargs == {"extra": {"extra_param": "value"}}
    
    def test_logging_calls_without_kwargs(self):
        """Test logging methods without extra kwargs."""
        mock_logger = Mock(spec=logging.Logger)
        handler = LoggingCallbackHandler(logger=mock_logger)
        
        # Test calls without kwargs to cover lines 90, 97, 104
        handler.on_stage_start("test_stage")
        handler.on_stage_end("test_stage")
        handler.on_stage_progress("test_stage", "progress message")
        
        # Check that info was called with correct messages
        assert mock_logger.info.call_count == 3
        
        calls = mock_logger.info.call_args_list
        assert calls[0].args == ("Stage start: test_stage",)
        assert calls[1].args == ("Stage end: test_stage",)
        assert calls[2].args == ("[test_stage] progress message",)


class TestCompositeCallbackHandler:
    """Test CompositeCallbackHandler."""
    
    def test_instantiation(self):
        """Test CompositeCallbackHandler instantiation."""
        handlers = [DefaultCallbackHandler(), DefaultCallbackHandler()]
        composite = CompositeCallbackHandler(handlers)
        assert composite.handlers == handlers
        assert len(composite.handlers) == 2
    
    def test_empty_handlers_list(self):
        """Test with empty handlers list."""
        composite = CompositeCallbackHandler([])
        assert composite.handlers == []
        
        # Should not raise exceptions
        composite.on_stage_start("test")
        composite.on_stage_end("test")
        composite.on_stage_progress("test", "message")
    
    def test_delegates_to_all_handlers(self):
        """Test that composite delegates to all handlers."""
        handler1 = Mock(spec=ProgressCallbackHandler)
        handler2 = Mock(spec=ProgressCallbackHandler)
        composite = CompositeCallbackHandler([handler1, handler2])
        
        composite.on_stage_start("test_stage", extra_param="value")
        composite.on_stage_end("test_stage", extra_param="value")
        composite.on_stage_progress("test_stage", "message", extra_param="value")
        
        # Both handlers should be called
        handler1.on_stage_start.assert_called_once_with("test_stage", extra_param="value")
        handler1.on_stage_end.assert_called_once_with("test_stage", extra_param="value")
        handler1.on_stage_progress.assert_called_once_with("test_stage", "message", extra_param="value")
        
        handler2.on_stage_start.assert_called_once_with("test_stage", extra_param="value")
        handler2.on_stage_end.assert_called_once_with("test_stage", extra_param="value")
        handler2.on_stage_progress.assert_called_once_with("test_stage", "message", extra_param="value")
    
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_handles_handler_exceptions(self, mock_stderr):
        """Test that exceptions in one handler don't break others."""
        failing_handler = Mock(spec=ProgressCallbackHandler)
        failing_handler.on_stage_start.side_effect = Exception("Handler failed")
        
        working_handler = Mock(spec=ProgressCallbackHandler)
        
        composite = CompositeCallbackHandler([failing_handler, working_handler])
        
        # Should not raise exception, but should continue to working handler
        composite.on_stage_start("test_stage")
        
        # Working handler should still be called
        working_handler.on_stage_start.assert_called_once_with("test_stage")
        
        # Should print warning to stderr
        output = mock_stderr.getvalue()
        assert "Warning: Callback handler failed: Handler failed" in output
    
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_handles_end_and_progress_exceptions(self, mock_stderr):
        """Test exception handling for on_stage_end and on_stage_progress."""
        failing_handler = Mock(spec=ProgressCallbackHandler)
        failing_handler.on_stage_end.side_effect = Exception("End failed")
        failing_handler.on_stage_progress.side_effect = Exception("Progress failed")
        
        composite = CompositeCallbackHandler([failing_handler])
        
        # This covers lines 132-133 (on_stage_end exception)
        composite.on_stage_end("test_stage")
        
        # This covers lines 140-141 (on_stage_progress exception) 
        composite.on_stage_progress("test_stage", "message")
        
        # Should print warnings to stderr
        output = mock_stderr.getvalue()
        assert "Warning: Callback handler failed: End failed" in output
        assert "Warning: Callback handler failed: Progress failed" in output
    
    def test_mixed_handler_types(self):
        """Test composite with different handler types."""
        cli_handler = CLICallbackHandler(verbose=False)  # Quiet to avoid output
        default_handler = DefaultCallbackHandler()
        mock_handler = Mock(spec=ProgressCallbackHandler)
        
        composite = CompositeCallbackHandler([cli_handler, default_handler, mock_handler])
        
        composite.on_stage_start("test_stage")
        composite.on_stage_end("test_stage")
        composite.on_stage_progress("test_stage", "message")
        
        # Mock handler should be called
        mock_handler.on_stage_start.assert_called_once_with("test_stage")
        mock_handler.on_stage_end.assert_called_once_with("test_stage")
        mock_handler.on_stage_progress.assert_called_once_with("test_stage", "message")


class TestIntegration:
    """Integration tests for callback system."""
    
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_real_workflow_simulation(self, mock_stderr):
        """Test simulating a real workflow with callbacks."""
        handler = CLICallbackHandler(verbose=True)
        
        # Simulate a three-stage workflow
        handler.on_stage_start("outline_generation")
        handler.on_stage_progress("outline_generation", "Analyzing requirements...")
        handler.on_stage_progress("outline_generation", "Generating document structure...")
        handler.on_stage_end("outline_generation")
        
        handler.on_stage_start("content_filling")
        handler.on_stage_progress("content_filling", "Filling section 1/3...")
        handler.on_stage_progress("content_filling", "Filling section 2/3...")
        handler.on_stage_progress("content_filling", "Filling section 3/3...")
        handler.on_stage_end("content_filling")
        
        handler.on_stage_start("review_revision")
        handler.on_stage_progress("review_revision", "Reviewing generated content...")
        handler.on_stage_progress("review_revision", "No issues found")
        handler.on_stage_end("review_revision")
        
        output = mock_stderr.getvalue()
        
        # Check all stages are present
        assert "outline_generation" in output
        assert "content_filling" in output
        assert "review_revision" in output
        
        # Check progress messages
        assert "Analyzing requirements..." in output
        assert "Filling section 1/3..." in output
        assert "No issues found" in output
    
    def test_composite_with_logging_and_cli(self):
        """Test composite handler with both logging and CLI handlers."""
        mock_logger = Mock(spec=logging.Logger)
        logging_handler = LoggingCallbackHandler(logger=mock_logger)
        cli_handler = CLICallbackHandler(verbose=False)  # Quiet
        
        composite = CompositeCallbackHandler([logging_handler, cli_handler])
        
        # Simulate workflow
        composite.on_stage_start("test_stage", workflow_id="123")
        composite.on_stage_progress("test_stage", "Working...", progress=50)
        composite.on_stage_end("test_stage", duration=1.5)
        
        # Check logging handler was called with extra context
        assert mock_logger.info.call_count == 3
        
        calls = mock_logger.info.call_args_list
        assert calls[0].kwargs == {"extra": {"workflow_id": "123"}}
        assert calls[1].kwargs == {"extra": {"progress": 50}}
        assert calls[2].kwargs == {"extra": {"duration": 1.5}}