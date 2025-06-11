"""DocuForge - AI-powered content rewrite engine for PRD documents."""

__version__ = "0.1.0"
__author__ = "DocuForge Team"
__email__ = "team@docuforge.ai"

from .callbacks import (
    CLICallbackHandler,
    CompositeCallbackHandler,
    DefaultCallbackHandler,
    LoggingCallbackHandler,
    ProgressCallbackHandler,
)
from .components import (
    ComponentBase,
    ContentFiller,
    ContextBuilder,
    OutlineGenerator,
    Reviser,
    RevisionIssue,
    RevisionReport,
)
from .models import (
    ClarificationItem,
    DocumentSection,
    DocumentStructure,
    RewriteRequest,
    RewriteResult,
)
from .prompts import PromptManager, PromptType
from .chain import RewriteChain, create_rewrite_chain
from . import cli

__all__ = [
    # Models
    "ClarificationItem",
    "DocumentSection", 
    "DocumentStructure",
    "RewriteRequest",
    "RewriteResult",
    # Callbacks
    "CLICallbackHandler",
    "CompositeCallbackHandler", 
    "DefaultCallbackHandler",
    "LoggingCallbackHandler",
    "ProgressCallbackHandler",
    # Components
    "ComponentBase",
    "ContentFiller",
    "ContextBuilder", 
    "OutlineGenerator",
    "Reviser",
    "RevisionIssue",
    "RevisionReport",
    # Prompts
    "PromptManager",
    "PromptType",
    # Chain
    "RewriteChain",
    "create_rewrite_chain",
    # CLI
    "cli",
]