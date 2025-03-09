# PDF to Audiobook Converter

A command-line tool that converts PDF research papers into audiobooks.

## Overview

The PDF to Audiobook Converter is designed to transform academic research papers and other PDFs into engaging audiobooks using text-to-speech (TTS) technology. It extracts text from PDFs, optionally applies custom titles, and produces both audio and text outputs.

## Features

- **PDF Text Extraction:** Extracts text from PDF files with support for custom titles.
- **Multiple TTS Engines:** Supports both Kokoro TTS (default) and OpenAI TTS for audio synthesis.
- **Audiobook Conversion:** Converts extracted text to audio using configurable TTS voices, models, and speed.
- **Custom Output Directory:** Use the `--output-folder` option to specify a directory for saving both audio and text files.
- **Centralized Logging:** Consistent logging across modules to aid in debugging and monitoring.
- **Flexible Command-Line Interface:** Multiple options for configuring the conversion process.
- **Web-based Interface:** A browser-based GUI for easy interaction without command-line knowledge.
- **Comprehensive Testing:** Extensive tests ensure reliability across a broad range of use cases.

## Installation

This project uses `uv` for dependency management. Please ensure you have `uv` installed and follow these steps:

1. Update dependencies in the `pyproject.toml` as needed.
2. Run the following command to sync dependencies:

```bash
uv sync
```

### Web GUI Requirements

The web-based GUI uses Gradio, which can be installed with:

```bash
uv pip install 'pdf-to-audiobook[web]'
```

Or directly:

```bash
uv pip install gradio
```

## Usage

### Command-Line Interface

Run the converter via the command line:

```bash
python -m pdf_to_audiobook.main [pdf_path] [options]
```

#### Command-Line Options

- `pdf_path`: Path to the PDF file.
- `--output-folder, -o`: Specifies the folder where generated files (audio and text) will be saved. If not provided, files are saved in their default locations.
- `--tts-mode`: TTS engine to use ('kokoro' or 'openai'). Default is `kokoro`.
- `--voice`: TTS voice to use. For OpenAI: 'alloy', 'nova', etc. For Kokoro: 'af_sky', etc. Default depends on the TTS mode.
- `--model`: TTS model to use (OpenAI only, e.g., 'tts-1', 'tts-1-hd'). Default is `tts-1`.
- `--speed`: Speech speed. Default is `1.0` for OpenAI and `1.25` for Kokoro.
- `--min-chunk-size`: Minimum chunk size for TTS processing.
- `--ai-model`: AI model for PDF text extraction (e.g., 'gemini', 'openai'). Default is `gemini`.
- `--title`: Custom title for the output files (this will be converted to snake_case).

#### Example

To convert a PDF using Kokoro TTS:

```bash
python -m pdf_to_audiobook.main research_paper.pdf --output-folder output --title 'Custom Paper Title' --tts-mode kokoro --voice af_sky --speed 1.25
```

To convert a PDF using OpenAI TTS:

```bash
python -m pdf_to_audiobook.main research_paper.pdf --output-folder output --title 'Custom Paper Title' --tts-mode openai --voice nova --model tts-1 --speed 1.0
```

### Web-based Interface

For a user-friendly experience, you can use the web-based GUI:

```bash
python -m pdf_to_audiobook.web_gui
```

This will start a local web server and open a browser window with the interface. The web GUI provides easy access to all features:

- PDF file upload via drag-and-drop or file browser
- Output folder and custom title options
- Voice, TTS Model, and Speed settings
- Advanced settings like Minimum Chunk Size and AI Model selection
- Status updates and progress tracking
- Audio playback directly in the browser

Once the interface loads, simply upload your PDF, configure your preferences, and click the "Convert PDF to Audiobook" button to start the conversion.

For quick access, you can also use:

```bash
python run_web_gui.py
```

## TTS Engines

### Kokoro TTS (Default)

An offline Text-to-Speech system with high-quality voices:
- Language support: American English (default), British English, Japanese, and Mandarin Chinese
- Voice options: Various voices such as 'af_sky' (default)
- Sample rate: 24000 Hz by default
- Speed range: Adjustable (1.25 by default)

### OpenAI TTS

OpenAI's cloud-based Text-to-Speech system:
- Voices: alloy, echo, fable, onyx, nova, shimmer
- Models: tts-1, tts-1-hd
- Speed range: 0.25 to 4.0
- Output formats: mp3, opus, aac, flac

## Environment Variables

Configure the tool by setting environment variables in a `.env` file:

```
# TTS Mode (openai)
TTS_MODE=openai

# OpenAI TTS configuration
OPENAI_API_KEY=your_openai_api_key
TTS_MODEL=tts-1
DEFAULT_VOICE=alloy
DEFAULT_SPEED=1.0
```

## Testing

Run the test suite using `pytest`:

```bash
python -m pytest -v
```

## Code Structure

- **pdf_to_audiobook/**: Main package containing modules for PDF reading, TTS conversion, logging, and utility functions.
  - **main.py**: Command-line interface and core conversion logic.
  - **web_gui.py**: Web-based graphical user interface using Gradio.
  - **tts_client.py**: Client for text-to-speech conversion.
  - **pdf_reader.py**: Functions to extract text from PDFs.
  - **config.py**: Configuration settings.
  - **file_utils.py**: Utilities for file operations.
  - **utils.py**: General utility functions.
  - **logging_config.py**: Logging configuration.
  - **validation.py**: Input validation functions.
- **tests/**: Contains unit tests using `pytest` to ensure the reliability of each module.

## Contributing

Contributions are welcome! Please adhere to the following guidelines:

- **Formatting:** Use Ruff and Black for code formatting.
- **Indentation:** 2 spaces as specified in the pyproject.toml.
- **Max Line Length:** 88 characters.
- **Naming Conventions:** snake_case for functions/variables, PascalCase for classes, and UPPER_CASE for constants.

## License

[Specify License Information Here] 