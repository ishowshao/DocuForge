"""Progress callback system for DocuForge rewrite engine."""

import sys
from abc import ABC, abstractmethod
from typing import Any, Optional


class ProgressCallbackHandler(ABC):
    """Abstract base class for progress callback handlers."""
    
    @abstractmethod
    def on_stage_start(self, stage_name: str, **kwargs: Any) -> None:
        """Called when a stage starts."""
        pass
    
    @abstractmethod
    def on_stage_end(self, stage_name: str, **kwargs: Any) -> None:
        """Called when a stage ends."""
        pass
    
    @abstractmethod
    def on_stage_progress(self, stage_name: str, message: str, **kwargs: Any) -> None:
        """Called to report progress within a stage."""
        pass


class DefaultCallbackHandler(ProgressCallbackHandler):
    """Default callback handler that does nothing."""
    
    def on_stage_start(self, stage_name: str, **kwargs: Any) -> None:
        """No-op implementation."""
        pass
    
    def on_stage_end(self, stage_name: str, **kwargs: Any) -> None:
        """No-op implementation.""" 
        pass
    
    def on_stage_progress(self, stage_name: str, message: str, **kwargs: Any) -> None:
        """No-op implementation."""
        pass


class CLICallbackHandler(ProgressCallbackHandler):
    """CLI callback handler that prints progress to stderr."""
    
    def __init__(self, verbose: bool = True):
        """Initialize CLI callback handler.
        
        Args:
            verbose: Whether to print detailed progress messages
        """
        self.verbose = verbose
    
    def on_stage_start(self, stage_name: str, **kwargs: Any) -> None:
        """Print stage start message."""
        if self.verbose:
            print(f"INFO: Stage start: {stage_name}...", file=sys.stderr)
    
    def on_stage_end(self, stage_name: str, **kwargs: Any) -> None:
        """Print stage end message."""
        if self.verbose:
            print(f"INFO: Stage end: {stage_name}.", file=sys.stderr)
    
    def on_stage_progress(self, stage_name: str, message: str, **kwargs: Any) -> None:
        """Print progress message."""
        if self.verbose:
            print(f"INFO: [{stage_name}] {message}", file=sys.stderr)


class LoggingCallbackHandler(ProgressCallbackHandler):
    """Callback handler that logs progress messages."""
    
    def __init__(self, logger: Optional[Any] = None):
        """Initialize logging callback handler.
        
        Args:
            logger: Logger instance to use (defaults to creating one)
        """
        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
    
    def on_stage_start(self, stage_name: str, **kwargs: Any) -> None:
        """Log stage start message."""
        if kwargs:
            self.logger.info(f"Stage start: {stage_name}", extra=kwargs)
        else:
            self.logger.info(f"Stage start: {stage_name}")
    
    def on_stage_end(self, stage_name: str, **kwargs: Any) -> None:
        """Log stage end message."""
        if kwargs:
            self.logger.info(f"Stage end: {stage_name}", extra=kwargs)
        else:
            self.logger.info(f"Stage end: {stage_name}")
    
    def on_stage_progress(self, stage_name: str, message: str, **kwargs: Any) -> None:
        """Log progress message."""
        if kwargs:
            self.logger.info(f"[{stage_name}] {message}", extra=kwargs)
        else:
            self.logger.info(f"[{stage_name}] {message}")


class CompositeCallbackHandler(ProgressCallbackHandler):
    """Composite callback handler that delegates to multiple handlers."""
    
    def __init__(self, handlers: list[ProgressCallbackHandler]):
        """Initialize composite callback handler.
        
        Args:
            handlers: List of callback handlers to delegate to
        """
        self.handlers = handlers
    
    def on_stage_start(self, stage_name: str, **kwargs: Any) -> None:
        """Delegate to all handlers."""
        for handler in self.handlers:
            try:
                handler.on_stage_start(stage_name, **kwargs)
            except Exception as e:
                # Don't let one handler's failure break others
                print(f"Warning: Callback handler failed: {e}", file=sys.stderr)
    
    def on_stage_end(self, stage_name: str, **kwargs: Any) -> None:
        """Delegate to all handlers."""
        for handler in self.handlers:
            try:
                handler.on_stage_end(stage_name, **kwargs)
            except Exception as e:
                print(f"Warning: Callback handler failed: {e}", file=sys.stderr)
    
    def on_stage_progress(self, stage_name: str, message: str, **kwargs: Any) -> None:
        """Delegate to all handlers."""
        for handler in self.handlers:
            try:
                handler.on_stage_progress(stage_name, message, **kwargs)
            except Exception as e:
                print(f"Warning: Callback handler failed: {e}", file=sys.stderr)