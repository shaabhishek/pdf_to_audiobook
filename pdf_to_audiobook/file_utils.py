"""File utility functions for the PDF to Audiobook converter.

This module provides common file operations used across the application.
"""

import os
from typing import Optional

from pdf_to_audiobook.logging_config import configure_logging

logger = configure_logging(__name__)


def ensure_directory_exists(directory_path: str) -> bool:
  """Ensure that a directory exists, creating it if necessary.

  Args:
      directory_path: Path to the directory to ensure exists.

  Returns:
      True if the directory exists or was created successfully, False otherwise.
  """
  try:
    os.makedirs(directory_path, exist_ok=True)
    return True
  except Exception as e:
    logger.error(f'Error creating directory {directory_path}: {e}')
    return False


def save_text_to_file(
  text: str, file_path: str, encoding: str = 'utf-8'
) -> Optional[str]:
  """Save text content to a file.

  Args:
      text: The text content to save.
      file_path: Path where the file should be saved.
      encoding: The encoding to use for the file.

  Returns:
      The absolute path to the saved file if successful, None otherwise.
  """
  try:
    # Ensure the directory exists
    directory = os.path.dirname(os.path.abspath(file_path))
    if not ensure_directory_exists(directory):
      return None

    with open(file_path, 'w', encoding=encoding) as f:
      f.write(text)

    return os.path.abspath(file_path)
  except Exception as e:
    logger.error(f'Error saving text to file {file_path}: {e}')
    return None


def save_binary_to_file(data: bytes, file_path: str) -> Optional[str]:
  """Save binary data to a file.

  Args:
      data: The binary data to save.
      file_path: Path where the file should be saved.

  Returns:
      The absolute path to the saved file if successful, None otherwise.
  """
  try:
    # Ensure the directory exists
    directory = os.path.dirname(os.path.abspath(file_path))
    if not ensure_directory_exists(directory):
      return None

    with open(file_path, 'wb') as f:
      f.write(data)

    return os.path.abspath(file_path)
  except Exception as e:
    logger.error(f'Error saving binary data to file {file_path}: {e}')
    return None
