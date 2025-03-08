"""Utility functions for the PDF to Audiobook converter.

This module provides utility functions for the PDF to Audiobook converter,
such as file naming and text processing.
"""

import os
import re

from pdf_to_audiobook.logging_config import configure_logging

# Configure logging
logger = configure_logging(__name__)


def to_snake_case(text: str) -> str:
  """Convert text to snake_case.

  Args:
      text: The text to convert.

  Returns:
      The text converted to snake_case.
  """
  if not text:
    return ''

  # Convert to lowercase
  text = text.lower()

  # Replace special characters with spaces
  text = re.sub(r'[^\w\s]', ' ', text)

  # Replace multiple spaces with a single space
  text = re.sub(r'\s+', ' ', text)

  # Replace spaces with underscores
  text = text.strip().replace(' ', '_')

  # Replace multiple underscores with a single underscore
  text = re.sub(r'_+', '_', text)

  return text


def get_snake_case_title_from_pdf(file_path: str) -> str:
  """Extract the title from a PDF and convert it to snake_case.

  Args:
      file_path: Path to the PDF file.

  Returns:
      The snake_case version of the PDF title, or the filename if extraction fails.
  """
  title = os.path.splitext(os.path.basename(file_path))[0]

  return to_snake_case(title)
