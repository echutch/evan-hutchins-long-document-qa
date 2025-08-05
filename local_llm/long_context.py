#!/usr/bin/env python3
"""
Local Long-Context LLM Testbed
A command-line tool for testing LLMs with long PDF documents via Ollama API.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import requests


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments containing pdf, prompt, and model.
    """
    parser = argparse.ArgumentParser(
        description="Test LLMs with long PDF documents via Ollama API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file to process"
    )
    
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to send to the model"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name to use (e.g., 'llama2', 'mistral')"
    )
    
    return parser.parse_args()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a valid PDF or is corrupted
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # Extract text from all pages
        text_content = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text_content += page.get_text()
            text_content += "\n"  # Add newline between pages
        
        doc.close()
        
        if not text_content.strip():
            raise ValueError(f"No text content found in PDF: {pdf_path}")
        
        return text_content.strip()
        
    except fitz.FileDataError as e:
        raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}") from e
    except FileNotFoundError:
        # Re-raise FileNotFoundError as-is
        raise
    except Exception as e:
        raise ValueError(f"Error processing PDF file {pdf_path}: {str(e)}") from e


def query_ollama(model: str, context: str, prompt: str) -> str:
    """
    Query the Ollama API with the given model, context, and prompt.
    
    Args:
        model (str): The Ollama model name
        context (str): The PDF text content to use as context
        prompt (str): The user's prompt/question
        
    Returns:
        str: The model's response
        
    Raises:
        ConnectionError: If unable to connect to Ollama API
        requests.RequestException: If API request fails
    """
    ollama_url = "http://localhost:11434/api/generate"
    
    # Construct the full prompt with context
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(
            ollama_url,
            json=payload,
            timeout=300  # 5 minute timeout for long contexts
        )
        
        # Check if request was successful
        if response.status_code != 200:
            raise requests.RequestException(
                f"Ollama API returned status {response.status_code}: {response.text}"
            )
        
        # Parse the JSON response
        response_data = response.json()
        
        if "response" not in response_data:
            raise ValueError(f"Unexpected response format from Ollama: {response_data}")
        
        return response_data["response"]
        
    except requests.ConnectionError as e:
        raise ConnectionError(
            "Unable to connect to Ollama API. Is Ollama running on localhost:11434?"
        ) from e
    except requests.Timeout as e:
        raise requests.RequestException("Request to Ollama API timed out") from e
    except requests.RequestException as e:
        raise requests.RequestException(f"Error communicating with Ollama API: {str(e)}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from Ollama API: {response.text}") from e


def log_interaction(log_dir: str, full_input: str, output: str) -> None:
    """
    Log the interaction to a timestamped file.
    
    Args:
        log_dir (str): Directory to store log files
        full_input (str): The complete input sent to the model
        output (str): The model's response
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_{timestamp}.log"
    log_filepath = log_path / log_filename
    
    # Write the interaction to the log file
    try:
        with open(log_filepath, 'w', encoding='utf-8') as log_file:
            log_file.write("=== Local LLM Testbed Experiment Log ===\n")
            log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
            log_file.write("=" * 50 + "\n\n")
            
            log_file.write("INPUT:\n")
            log_file.write("-" * 20 + "\n")
            log_file.write(full_input)
            log_file.write("\n\n")
            
            log_file.write("OUTPUT:\n")
            log_file.write("-" * 20 + "\n")
            log_file.write(output)
            log_file.write("\n")
        
        print(f"Interaction logged to: {log_filepath}")
        
    except Exception as e:
        print(f"Warning: Failed to write log file: {e}", file=sys.stderr)


def main():
    """
    Main entry point of the application.
    Orchestrates the workflow: parse args -> extract PDF -> query API -> log results.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        print(f"Processing PDF: {args.pdf}")
        print(f"Using model: {args.model}")
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(args.pdf)
        print(f"Extracted {len(pdf_text)} characters from PDF")
        
        # Query Ollama API
        print("Querying Ollama API...")
        full_input = f"Context:\n{pdf_text}\n\nQuestion: {args.prompt}\n\nAnswer:"
        response = query_ollama(args.model, pdf_text, args.prompt)
        
        # Log the interaction
        print("Logging interaction...")
        log_interaction("logs", full_input, response)
        
        # Print the response
        print("\n" + "=" * 50)
        print("MODEL RESPONSE:")
        print("=" * 50)
        print(response)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as e:
        print(f"Connection Error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"API Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
