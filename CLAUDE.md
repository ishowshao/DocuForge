# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocuForge is an AI-powered content rewrite engine for PRD (Product Requirements Document) writing. The project implements a three-stage rewrite algorithm: "Structure First, Sequential Generation, Closed-loop Correction" using LangChain/LangGraph frameworks.

## Core Architecture

The system is designed around a **three-stage rewrite algorithm**:

1. **Stage 1**: Outline & Structure Initialization - Creates `DocumentStructure` with empty content sections
2. **Stage 2**: Sequential Content Filling - Fills sections using rolling context approach  
3. **Stage 3**: Review and Revision - AI-powered review with targeted fixes

Key components (from `docs/tech.md`):
- `RewriteChain` - Main orchestrator using LangGraph
- `ContextBuilder` - Aggregates original content + clarifications
- `OutlineGenerator` - Creates initial document structure
- `ContentFiller` - Sequential section generation with context continuity
- `Reviser` - AI review and targeted revision system

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Test Azure OpenAI integration
python example.py
```

### Planned CLI (not yet implemented)
```bash
python -m prd_writer.rewrite \
    --original-doc ./path/to/original_document.md \
    --clarifications ./path/to/clarifications.json \
    --output-md ./path/to/rewritten_document.md \
    --output-json ./path/to/structured_output.json
```

## Key Data Models

Based on `docs/tech.md`, core data structures:

```python
@dataclass
class RewriteRequest:
    original_content: str
    clarifications: List[ClarificationItem]

@dataclass  
class DocumentStructure:
    title: str
    sections: List[DocumentSection]
    metadata: Dict[str, Any]

@dataclass
class DocumentSection:
    title: str
    content: str
    level: int
    order: int
    goal: str  # Writing objective for the section
```

## Technology Stack

- **Python 3.10+** with LangChain ecosystem
- **LangChain 0.3.25** - LLM framework and prompt management
- **LangGraph 0.4.8** - Stateful agent workflows with conditional edges
- **LangSmith 0.3.45** - Debugging and observability
- **Pydantic 2.11.5** - Data validation and structured outputs
- **Azure OpenAI** - GPT-4 integration

## Environment Configuration

Required environment variables (in `.env`):
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME` 
- `AZURE_OPENAI_API_VERSION`

## Development Status

**Current State**: Early development with technical specification complete
- Core algorithm documented in `docs/tech.md` (259 lines)
- Basic Azure OpenAI integration example in `example.py`
- Dependencies configured but core implementation pending

**Missing Implementation**:
- Package structure (`src/` or `prd_writer/` directory)
- Core `RewriteChain` LangGraph implementation
- CLI module and argument parsing
- Test framework and test files
- Package configuration (`setup.py`/`pyproject.toml`)

## Implementation Priority

When implementing core functionality, follow this order:
1. Create package structure matching `docs/tech.md` specifications
2. Implement core data models with Pydantic validation
3. Build `RewriteChain` as LangGraph workflow
4. Add CLI interface with progress callbacks
5. Implement error handling and retry logic for LLM calls