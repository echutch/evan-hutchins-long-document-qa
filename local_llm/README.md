# Local Long-Context LLM Testbed

A command-line tool for testing LLMs with long PDF documents via Ollama API.

## Overview

This tool extracts text from PDF documents and sends it as context to local LLM models running via Ollama. It's designed for research and experimentation with long-context language models.

## Features

- PDF text extraction using PyMuPDF
- Integration with Ollama API
- Automatic logging of experiments
- Comprehensive error handling
- Unit tests for reliability

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and setup Ollama:**
   - Download from: https://ollama.ai/
   - Install a model, e.g.: `ollama pull llama2`

## Usage

```bash
python long_context.py --pdf <path_to_pdf> --prompt "<your_question>" --model <ollama_model>
```

### Example

```bash
python long_context.py --pdf sample_test.pdf --prompt "What is this document about?" --model llama2
```

### Arguments

- `--pdf`: Path to the PDF file to process (required)
- `--prompt`: Question or prompt to send to the model (required)
- `--model`: Ollama model name to use (required)

## Output

The tool will:
1. Extract text from the PDF
2. Send the text as context along with your prompt to the Ollama model
3. Display the model's response
4. Log the complete interaction to a timestamped file in the `logs/` directory

## Project Structure

```
local_llm/
├── long_context.py          # Main application script
├── test_experiment.py       # Unit tests
├── requirements.txt         # Project dependencies
├── sample_test.pdf         # Sample PDF for testing
└── README.md               # This file
```

## Testing

Run the unit tests:

```bash
python -m pytest test_experiment.py -v
```

## Dependencies

- **PyMuPDF (fitz)**: PDF text extraction
- **requests**: HTTP client for Ollama API
- **pytest**: Testing framework

## Error Handling

The tool provides clear error messages for common issues:
- PDF file not found
- Invalid or corrupted PDF files
- Ollama API connection problems
- Model not available
- API timeouts

## Logging

All interactions are automatically logged to timestamped files in the `logs/` directory with the format:
- Filename: `experiment_YYYYMMDD_HHMMSS.log`
- Contains: Full input context, prompt, and model response

## Requirements

- Python 3.10+
- Ollama running on localhost:11434
- At least one Ollama model installed

## Troubleshooting

**"Unable to connect to Ollama API"**
- Ensure Ollama is running: `ollama serve`
- Check if running on default port 11434

**"Model not found"**
- Install the model: `ollama pull <model_name>`
- List available models: `ollama list`

**PDF extraction issues**
- Ensure the PDF file exists and is readable
- Check that the PDF contains text (not just images)

## License

This project is designed for research and educational purposes.
