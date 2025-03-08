"""Validation utilities for PDF to Audiobook converter."""

import argparse
import os

from pdf_to_audiobook.logging_config import configure_logging

# Configure logging
logger = configure_logging(__name__)


def valid_pdf_file(pdf_path: str) -> str:
  """Validate that the given path points to a readable PDF file.

  Args:
    pdf_path: The path to the PDF file to validate.

  Returns:
    The validated PDF path.

  Raises:
    argparse.ArgumentTypeError: If the path is not a valid, readable PDF file.
  """
  # Check if file exists
  if not os.path.exists(pdf_path):
    raise argparse.ArgumentTypeError(f'File does not exist: {pdf_path}')

  # Check if it's a file
  if not os.path.isfile(pdf_path):
    raise argparse.ArgumentTypeError(f'Not a file: {pdf_path}')

  # Check if it has a PDF extension
  if not pdf_path.lower().endswith('.pdf'):
    raise argparse.ArgumentTypeError(f'Not a PDF file: {pdf_path}')

  # Check if file is readable
  if not os.access(pdf_path, os.R_OK):
    raise argparse.ArgumentTypeError(f'File is not readable: {pdf_path}')

  return pdf_path


def valid_speed(speed_str: str) -> float:
  """Validate the TTS speed parameter.

  Args:
    speed_str: The speed as a string to validate (0.25 to 4.0).

  Returns:
    The validated speed as a float.

  Raises:
    argparse.ArgumentTypeError: If the speed is not valid.
  """
  try:
    speed = float(speed_str)
  except ValueError:
    raise argparse.ArgumentTypeError(f'Invalid speed value: {speed_str}')

  if speed < 0.25 or speed > 4.0:
    raise argparse.ArgumentTypeError('Speed must be between 0.25 and 4.0')

  return speed
