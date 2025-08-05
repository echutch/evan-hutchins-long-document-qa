#!/usr/bin/env python3
"""
Unit tests for the Local Long-Context LLM Testbed.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest
import requests

# Import the functions we want to test
from long_context import (
    parse_arguments,
    extract_text_from_pdf,
    query_ollama,
    log_interaction
)


class TestParseArguments(unittest.TestCase):
    """Test the argument parsing functionality."""
    
    @patch('sys.argv', ['long_context.py', '--pdf', 'test.pdf', '--prompt', 'What is this about?', '--model', 'llama2'])
    def test_parse_arguments_valid(self):
        """Test parsing valid arguments."""
        args = parse_arguments()
        self.assertEqual(args.pdf, 'test.pdf')
        self.assertEqual(args.prompt, 'What is this about?')
        self.assertEqual(args.model, 'llama2')
    
    @patch('sys.argv', ['long_context.py', '--pdf', 'test.pdf'])
    def test_parse_arguments_missing_required(self):
        """Test that missing required arguments raise SystemExit."""
        with self.assertRaises(SystemExit):
            parse_arguments()


class TestExtractTextFromPDF(unittest.TestCase):
    """Test the PDF text extraction functionality."""
    
    def test_extract_text_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf("nonexistent.pdf")
    
    @patch('fitz.open')
    def test_extract_text_valid_pdf(self, mock_fitz_open):
        """Test successful text extraction from a valid PDF."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 2
        
        # Mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content"
        
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            result = extract_text_from_pdf("test.pdf")
            
        expected = "Page 1 content\nPage 2 content"
        self.assertEqual(result, expected)
        mock_doc.close.assert_called_once()
    
    @patch('fitz.open')
    def test_extract_text_empty_pdf(self, mock_fitz_open):
        """Test that ValueError is raised for PDFs with no text content."""
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        with patch('os.path.exists', return_value=True):
            with self.assertRaises(ValueError):
                extract_text_from_pdf("empty.pdf")
    
    @patch('fitz.open')
    def test_extract_text_corrupted_pdf(self, mock_fitz_open):
        """Test that ValueError is raised for corrupted PDF files."""
        mock_fitz_open.side_effect = Exception("FileDataError: corrupted file")
        
        with patch('os.path.exists', return_value=True):
            with self.assertRaises(ValueError):
                extract_text_from_pdf("corrupted.pdf")


class TestQueryOllama(unittest.TestCase):
    """Test the Ollama API interaction functionality."""
    
    @patch('requests.post')
    def test_query_ollama_success(self, mock_post):
        """Test successful API query."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response from model"}
        mock_post.return_value = mock_response
        
        result = query_ollama("llama2", "Test context", "Test prompt")
        
        self.assertEqual(result, "Test response from model")
        
        # Verify the API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json']['model'], 'llama2')
        self.assertIn('Test context', call_args[1]['json']['prompt'])
        self.assertIn('Test prompt', call_args[1]['json']['prompt'])
    
    @patch('requests.post')
    def test_query_ollama_connection_error(self, mock_post):
        """Test handling of connection errors."""
        mock_post.side_effect = requests.ConnectionError("Connection failed")
        
        with self.assertRaises(ConnectionError):
            query_ollama("llama2", "Test context", "Test prompt")
    
    @patch('requests.post')
    def test_query_ollama_non_200_status(self, mock_post):
        """Test handling of non-200 HTTP status codes."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        with self.assertRaises(requests.RequestException):
            query_ollama("llama2", "Test context", "Test prompt")
    
    @patch('requests.post')
    def test_query_ollama_invalid_json(self, mock_post):
        """Test handling of invalid JSON responses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        mock_post.return_value = mock_response
        
        with self.assertRaises(ValueError):
            query_ollama("llama2", "Test context", "Test prompt")
    
    @patch('requests.post')
    def test_query_ollama_missing_response_field(self, mock_post):
        """Test handling of responses missing the 'response' field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "No response field"}
        mock_post.return_value = mock_response
        
        with self.assertRaises(ValueError):
            query_ollama("llama2", "Test context", "Test prompt")


class TestLogInteraction(unittest.TestCase):
    """Test the logging functionality."""
    
    def test_log_interaction_creates_file(self):
        """Test that log_interaction creates a log file with correct content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_input = "Test input content"
            test_output = "Test output content"
            
            log_interaction(temp_dir, test_input, test_output)
            
            # Check that a log file was created
            log_files = list(Path(temp_dir).glob("experiment_*.log"))
            self.assertEqual(len(log_files), 1)
            
            # Check file content
            with open(log_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.assertIn("Local LLM Testbed Experiment Log", content)
            self.assertIn("INPUT:", content)
            self.assertIn("OUTPUT:", content)
            self.assertIn(test_input, content)
            self.assertIn(test_output, content)
    
    def test_log_interaction_creates_directory(self):
        """Test that log_interaction creates the log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "nonexistent_logs")
            
            log_interaction(log_dir, "test input", "test output")
            
            # Check that the directory was created
            self.assertTrue(os.path.exists(log_dir))
            
            # Check that a log file was created
            log_files = list(Path(log_dir).glob("experiment_*.log"))
            self.assertEqual(len(log_files), 1)


if __name__ == "__main__":
    unittest.main()
