# PDF to Audiobook Converter

A Python library that converts PDF research papers into audiobooks (MP3 format).

## Features

- Extract text from PDF research papers using Google's Gemini API
- Convert extracted text to speech using OpenAI's TTS API
- Save the result as an MP3 file in the same directory as the input PDF (by default)
- Command-line interface for easy use

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pdf-to-audiobook.git
   cd pdf-to-audiobook
   ```

2. Install the package:
   ```
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install the package and dependencies
   uv sync
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini configuration (for PDF text extraction)
GEMINI_MODEL=gemini-1.5-pro

# OpenAI TTS configuration (for text-to-speech)
TTS_MODEL=tts-1                # Options: tts-1, tts-1-hd
DEFAULT_VOICE=alloy            # Options: alloy, echo, fable, onyx, nova, shimmer
DEFAULT_SPEED=1.0              # Speed multiplier (0.25 to 4.0)
```

## Usage

### Command Line

```
python -m pdf_to_audiobook.main path/to/research_paper.pdf
```

By default, the output MP3 file will be saved to the same directory as the input PDF file, with the same name but with the AI model name appended and a .mp3 extension. For example, if you convert `/path/to/research_paper.pdf` using the Gemini AI model, the output will be saved as `/path/to/research_paper_gemini.mp3`.

Options:
- `--output`, `-o`: Custom path to save the output MP3 file (optional)
- `--voice`, `-v`: Voice to use for the audiobook (default: alloy)
- `--model`, `-m`: TTS model to use (default: tts-1)
- `--speed`, `-s`: Speech speed multiplier (default: 1.0)
- `--ai-model`: AI model to use for PDF processing ('gemini' or 'openai', default: gemini)

Example:
```
python -m pdf_to_audiobook.main research_paper.pdf --voice nova --model tts-1-hd --speed 1.2 --ai-model openai
```

### Python API

```python
from pdf_to_audiobook.main import convert_pdf_to_audiobook

success = convert_pdf_to_audiobook(
    pdf_path='research_paper.pdf',
    output_path=None,  # None uses the same path/name as PDF but with AI model name and .mp3 extension
    voice='nova',      # OpenAI voice
    model='tts-1-hd',  # OpenAI TTS model
    speed=1.2,         # Speed multiplier
    ai_model='openai'  # AI model used for PDF processing (default: 'openai')
)

if success:
    print('Conversion successful!')
else:
    print('Conversion failed.')
```

## Requirements

- Python 3.10+
- Google Gemini API key (for text extraction)
- OpenAI API key (for TTS)

## Development

### Package Management with uv

This project uses [uv](https://github.com/astral-sh/uv) for Python package management. To add or modify dependencies:

1. Edit the `pyproject.toml` file to add or update dependencies
2. Run `uv sync` to install the updated dependencies
3. Use `uv sync --all-extras` to install development dependencies as well

### Running Tests

To run the tests, install the development dependencies and use pytest:

```
# Install the package with development dependencies
uv sync --all-extras

# Run the tests
uv run pytest
```

The test suite includes unit tests for all components of the package.

## License

MIT 