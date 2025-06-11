"""Unit tests for CLI module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docuforge.cli import (
    create_argument_parser,
    load_clarifications_from_file,
    load_document_from_file,
    main,
    save_output_file,
    setup_llm,
)
from docuforge.models import ClarificationItem


class TestLoadClarificationsFromFile:
    """Test clarifications loading functionality."""
    
    def test_load_valid_clarifications_array(self):
        """Test loading valid clarifications from array format."""
        clarifications_data = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(clarifications_data, f)
            temp_file = f.name
        
        try:
            result = load_clarifications_from_file(temp_file)
            
            assert len(result) == 2
            assert isinstance(result[0], ClarificationItem)
            assert result[0].question == "Q1?"
            assert result[0].answer == "A1"
            assert result[1].question == "Q2?"
            assert result[1].answer == "A2"
        finally:
            os.unlink(temp_file)
    
    def test_load_valid_clarifications_object(self):
        """Test loading valid clarifications from object format."""
        clarifications_data = {
            "clarifications": [
                {"question": "Q1?", "answer": "A1"},
                {"question": "Q2?", "answer": "A2"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(clarifications_data, f)
            temp_file = f.name
        
        try:
            result = load_clarifications_from_file(temp_file)
            
            assert len(result) == 2
            assert result[0].question == "Q1?"
            assert result[1].question == "Q2?"
        finally:
            os.unlink(temp_file)
    
    def test_load_clarifications_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Clarifications file not found"):
            load_clarifications_from_file("nonexistent.json")
    
    def test_load_clarifications_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid JSON format"):
                load_clarifications_from_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_clarifications_missing_fields(self):
        """Test loading clarifications with missing required fields."""
        clarifications_data = [
            {"question": "Q1?"},  # Missing answer
            {"answer": "A2"}      # Missing question
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(clarifications_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="must have 'question' and 'answer' fields"):
                load_clarifications_from_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_clarifications_invalid_format(self):
        """Test loading clarifications with invalid format."""
        invalid_data = "not an array or object"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid clarifications format"):
                load_clarifications_from_file(temp_file)
        finally:
            os.unlink(temp_file)


class TestLoadDocumentFromFile:
    """Test document loading functionality."""
    
    def test_load_valid_document(self):
        """Test loading valid document."""
        content = "This is a test document content."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            result = load_document_from_file(temp_file)
            assert result == content
        finally:
            os.unlink(temp_file)
    
    def test_load_document_with_whitespace(self):
        """Test loading document with leading/trailing whitespace."""
        content = "  \n  Document content  \n  "
        expected = "Document content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            result = load_document_from_file(temp_file)
            assert result == expected
        finally:
            os.unlink(temp_file)
    
    def test_load_document_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Document file not found"):
            load_document_from_file("nonexistent.md")
    
    def test_load_empty_document(self):
        """Test loading empty document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Document file is empty"):
                load_document_from_file(temp_file)
        finally:
            os.unlink(temp_file)


class TestSaveOutputFile:
    """Test output file saving functionality."""
    
    def test_save_output_file(self):
        """Test saving output to file."""
        content = "Test output content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_file = f.name
        
        try:
            os.unlink(temp_file)  # Remove the file so we can test creation
            save_output_file(content, temp_file)
            
            # Verify file was created and content is correct
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                saved_content = f.read()
            assert saved_content == content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_save_output_file_create_directory(self):
        """Test saving output file with directory creation."""
        content = "Test content"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "output.md")
            
            save_output_file(content, output_path)
            
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                saved_content = f.read()
            assert saved_content == content
    
    def test_save_output_file_invalid_path(self):
        """Test saving to invalid path."""
        with pytest.raises(ValueError, match="Error writing output file"):
            # Try to write to a directory that can't be created (like root on most systems)
            save_output_file("content", "/invalid/path/that/cannot/exist/file.txt")


class TestSetupLLM:
    """Test LLM setup functionality."""
    
    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        "AZURE_OPENAI_API_VERSION": "2023-12-01-preview"
    })
    @patch('docuforge.cli.AzureChatOpenAI')
    def test_setup_llm_success(self, mock_azure_openai):
        """Test successful LLM setup."""
        mock_llm = Mock()
        mock_azure_openai.return_value = mock_llm
        
        result = setup_llm()
        
        assert result == mock_llm
        mock_azure_openai.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="test-deployment",
            openai_api_version="2023-12-01-preview",
            reasoning_effort="high"
        )
    
    @patch('docuforge.cli.load_dotenv')
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_llm_missing_env_vars(self, mock_load_dotenv):
        """Test LLM setup with missing environment variables."""
        with pytest.raises(ValueError, match="Missing required environment variables"):
            setup_llm()
    
    @patch('docuforge.cli.load_dotenv')
    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment"
        # Missing AZURE_OPENAI_API_VERSION
    }, clear=True)
    def test_setup_llm_partial_env_vars(self, mock_load_dotenv):
        """Test LLM setup with partial environment variables."""
        with pytest.raises(ValueError, match="Missing required environment variables"):
            setup_llm()
    
    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        "AZURE_OPENAI_API_VERSION": "2023-12-01-preview"
    })
    @patch('docuforge.cli.AzureChatOpenAI')
    def test_setup_llm_initialization_error(self, mock_azure_openai):
        """Test LLM setup with initialization error."""
        mock_azure_openai.side_effect = Exception("Initialization failed")
        
        with pytest.raises(ValueError, match="Failed to initialize Azure OpenAI client"):
            setup_llm()


class TestArgumentParser:
    """Test argument parser functionality."""
    
    def test_create_argument_parser(self):
        """Test creating argument parser."""
        parser = create_argument_parser()
        
        # Test required arguments
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Should fail due to missing required args
        
        # Test valid arguments
        args = parser.parse_args([
            "--original-doc", "test.md",
            "--clarifications", "clarify.json"
        ])
        
        assert args.original_doc == "test.md"
        assert args.clarifications == "clarify.json"
        assert args.max_revision_rounds == 3  # default
        assert args.quiet is False  # default
        assert args.output_md is None  # default
        assert args.output_json is None  # default
    
    def test_argument_parser_all_options(self):
        """Test argument parser with all options."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            "--original-doc", "test.md",
            "--clarifications", "clarify.json",
            "--output-md", "output.md",
            "--output-json", "output.json",
            "--max-revision-rounds", "5",
            "--quiet"
        ])
        
        assert args.original_doc == "test.md"
        assert args.clarifications == "clarify.json"
        assert args.output_md == "output.md"
        assert args.output_json == "output.json"
        assert args.max_revision_rounds == 5
        assert args.quiet is True


class TestMainFunction:
    """Test main CLI function."""
    
    @patch('docuforge.cli.create_rewrite_chain')
    @patch('docuforge.cli.setup_llm')
    @patch('docuforge.cli.load_document_from_file')
    @patch('docuforge.cli.load_clarifications_from_file')
    @patch('sys.argv', ['docuforge', '--original-doc', 'test.md', '--clarifications', 'clarify.json'])
    def test_main_success_stdout(self, mock_load_clarify, mock_load_doc, mock_setup_llm, mock_create_chain):
        """Test successful main execution with stdout output."""
        # Setup mocks
        mock_load_doc.return_value = "Test document content"
        mock_load_clarify.return_value = [
            ClarificationItem(question="Q?", answer="A")
        ]
        mock_llm = Mock()
        mock_setup_llm.return_value = mock_llm
        
        mock_result = Mock()
        mock_result.rewritten_content = "Rewritten content"
        mock_result.structured_document.model_dump_json.return_value = '{"test": "json"}'
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        mock_create_chain.return_value = mock_chain
        
        with patch('builtins.print') as mock_print, patch('sys.stderr'):
            result = main()
        
        assert result == 0
        # Check that rewritten content was printed to stdout
        stdout_calls = [call for call in mock_print.call_args_list if len(call.kwargs) == 0]
        assert any(call.args[0] == "Rewritten content" for call in stdout_calls)
    
    @patch('docuforge.cli.create_rewrite_chain')
    @patch('docuforge.cli.setup_llm')
    @patch('docuforge.cli.load_document_from_file')
    @patch('docuforge.cli.load_clarifications_from_file')
    @patch('docuforge.cli.save_output_file')
    @patch('sys.argv', [
        'docuforge', '--original-doc', 'test.md', '--clarifications', 'clarify.json',
        '--output-md', 'output.md', '--quiet'
    ])
    def test_main_success_file_output(self, mock_save, mock_load_clarify, mock_load_doc, mock_setup_llm, mock_create_chain):
        """Test successful main execution with file output."""
        # Setup mocks
        mock_load_doc.return_value = "Test document content"
        mock_load_clarify.return_value = []
        mock_llm = Mock()
        mock_setup_llm.return_value = mock_llm
        
        mock_result = Mock()
        mock_result.rewritten_content = "Rewritten content"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        mock_create_chain.return_value = mock_chain
        
        result = main()
        
        assert result == 0
        mock_save.assert_called_once_with("Rewritten content", "output.md")
    
    @patch('docuforge.cli.load_document_from_file')
    @patch('sys.argv', ['docuforge', '--original-doc', 'test.md', '--clarifications', 'clarify.json'])
    def test_main_file_not_found_error(self, mock_load_doc):
        """Test main with file not found error."""
        mock_load_doc.side_effect = FileNotFoundError("File not found")
        
        with patch('sys.stderr'):
            result = main()
        
        assert result == 1
    
    @patch('docuforge.cli.load_document_from_file')
    @patch('sys.argv', ['docuforge', '--original-doc', 'test.md', '--clarifications', 'clarify.json'])
    def test_main_keyboard_interrupt(self, mock_load_doc):
        """Test main with keyboard interrupt."""
        mock_load_doc.side_effect = KeyboardInterrupt()
        
        with patch('sys.stderr'):
            result = main()
        
        assert result == 1
    
    @patch('docuforge.cli.load_document_from_file')
    @patch('sys.argv', ['docuforge', '--original-doc', 'test.md', '--clarifications', 'clarify.json'])
    def test_main_general_error(self, mock_load_doc):
        """Test main with general error."""
        mock_load_doc.side_effect = ValueError("General error")
        
        with patch('sys.stderr'):
            result = main()
        
        assert result == 1


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_cli_help(self):
        """Test that CLI help works."""
        parser = create_argument_parser()
        
        # This should not raise an exception
        help_text = parser.format_help()
        assert "docuforge" in help_text
        assert "--original-doc" in help_text
        assert "--clarifications" in help_text
    
    def test_example_clarifications_format(self):
        """Test that example clarifications format is valid."""
        example_data = [
            {
                "question": "What is the main goal?",
                "answer": "To improve user experience"
            },
            {
                "question": "Who is the target audience?", 
                "answer": "Enterprise customers"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(example_data, f)
            temp_file = f.name
        
        try:
            result = load_clarifications_from_file(temp_file)
            assert len(result) == 2
            assert all(isinstance(item, ClarificationItem) for item in result)
        finally:
            os.unlink(temp_file)