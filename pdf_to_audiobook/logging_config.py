"""Centralized logging configuration for the PDF to Audiobook converter.

This module provides a consistent logging configuration across all modules.
"""

import logging
from typing import Optional

from pdf_to_audiobook.config import LOG_LEVEL


def configure_logging(name: Optional[str] = None) -> logging.Logger:
  """Configure and return a logger with consistent formatting.

  Args:
      name: The name for the logger. If None, returns the root logger.

  Returns:
      A configured logger instance.
  """
  # Configure the root logger if not already configured
  if not logging.getLogger().handlers:
    logging.basicConfig(
      level=getattr(logging, LOG_LEVEL),
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

  # Return the requested logger
  return logging.getLogger(name)
