"""PDF to Audiobook converter.

This package provides functionality to convert PDF research papers into audiobooks.
It extracts text from PDFs using OpenAI's API and converts it to speech using
a cloud-based TTS service. The package includes both a command-line interface
and a graphical user interface.
"""

__version__ = '0.1.0'

# Import main modules for easy access
from pdf_to_audiobook.main import convert_pdf_to_audiobook
