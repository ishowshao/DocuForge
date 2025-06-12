"""Command-line interface for DocuForge."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from .callbacks import CLICallbackHandler
from .chain import create_rewrite_chain
from .models import ClarificationItem, RewriteRequest


def load_clarifications_from_file(file_path: str) -> List[ClarificationItem]:
    """Load clarifications from JSON file.
    
    Args:
        file_path: Path to JSON file containing clarifications
        
    Returns:
        List of ClarificationItem objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Clarifications file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support both array format and object format
        if isinstance(data, list):
            clarifications_data = data
        elif isinstance(data, dict) and "clarifications" in data:
            clarifications_data = data["clarifications"]
        else:
            raise ValueError("Invalid clarifications format")
        
        clarifications = []
        for item in clarifications_data:
            if not isinstance(item, dict):
                raise ValueError("Each clarification must be an object")
            
            if "question" not in item or "answer" not in item:
                raise ValueError("Each clarification must have 'question' and 'answer' fields")
            
            clarifications.append(ClarificationItem(
                question=item["question"],
                answer=item["answer"]
            ))
        
        return clarifications
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in clarifications file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading clarifications: {e}")


def load_document_from_file(file_path: str) -> str:
    """Load document content from file.
    
    Args:
        file_path: Path to document file
        
    Returns:
        Document content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError("Document file is empty")
        
        return content
        
    except Exception as e:
        raise ValueError(f"Error reading document file: {e}")


def save_output_file(content: str, file_path: str) -> None:
    """Save content to output file.
    
    Args:
        content: Content to save
        file_path: Output file path
        
    Raises:
        ValueError: If file cannot be written
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception as e:
        raise ValueError(f"Error writing output file '{file_path}': {e}")


def setup_llm() -> AzureChatOpenAI:
    """Setup and configure the language model.
    
    Returns:
        Configured AzureChatOpenAI instance
        
    Raises:
        ValueError: If environment variables are not set
    """
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            "Please set them in your .env file or environment."
        )
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            # reasoning_effort="high"
        )
        return llm
        
    except Exception as e:
        raise ValueError(f"Failed to initialize Azure OpenAI client: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="docuforge",
        description="AI-powered content rewrite engine for PRD documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with original document and clarifications
  docuforge --original-doc document.md --clarifications clarify.json

  # Output to specific files
  docuforge --original-doc doc.md --clarifications clarify.json \\
            --output-md rewritten.md --output-json structure.json

  # Quiet mode (no progress output)
  docuforge --original-doc doc.md --clarifications clarify.json --quiet

Clarifications JSON format:
  [
    {
      "question": "What is the main goal?",
      "answer": "To improve user experience"
    },
    {
      "question": "Who is the target audience?",
      "answer": "Enterprise customers"
    }
  ]
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--original-doc",
        required=True,
        type=str,
        help="Path to the original document file to be rewritten"
    )
    
    parser.add_argument(
        "--clarifications",
        required=True,
        type=str,
        help="Path to JSON file containing clarification question-answer pairs"
    )
    
    # Optional output arguments
    parser.add_argument(
        "--output-md",
        type=str,
        help="Path to save the rewritten document in Markdown format"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save the structured document in JSON format"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--max-revision-rounds",
        type=int,
        default=3,
        help="Maximum number of revision rounds (default: 3)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup callback handler
    callback_handler = CLICallbackHandler(verbose=not args.quiet)
    
    try:
        # Load inputs
        if not args.quiet:
            print("Loading original document...", file=sys.stderr)
        original_content = load_document_from_file(args.original_doc)
        
        if not args.quiet:
            print("Loading clarifications...", file=sys.stderr)
        clarifications = load_clarifications_from_file(args.clarifications)
        
        # Create request
        request = RewriteRequest(
            original_content=original_content,
            clarifications=clarifications
        )
        
        # Setup LLM
        if not args.quiet:
            print("Initializing AI model...", file=sys.stderr)
        llm = setup_llm()
        
        # Create and execute rewrite chain
        if not args.quiet:
            print("Starting document rewrite process...", file=sys.stderr)
        
        chain = create_rewrite_chain(
            llm=llm,
            callback_handler=callback_handler,
            max_revision_rounds=args.max_revision_rounds
        )
        
        result = chain.invoke(request)
        
        # Handle outputs
        outputs_saved = False
        
        if args.output_md:
            if not args.quiet:
                print(f"Saving Markdown output to {args.output_md}...", file=sys.stderr)
            save_output_file(result.rewritten_content, args.output_md)
            outputs_saved = True
        
        if args.output_json:
            if not args.quiet:
                print(f"Saving JSON output to {args.output_json}...", file=sys.stderr)
            json_content = result.structured_document.model_dump_json(indent=2)
            save_output_file(json_content, args.output_json)
            outputs_saved = True
        
        # If no output files specified, print to stdout
        if not outputs_saved:
            print(result.rewritten_content)
        
        if not args.quiet:
            print("Document rewrite completed successfully!", file=sys.stderr)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())