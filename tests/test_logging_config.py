"""Tests for the logging_config module."""

import logging
import unittest
from unittest.mock import patch

from pdf_to_audiobook.logging_config import configure_logging


class TestLoggingConfig(unittest.TestCase):
  """Test cases for the logging_config module."""

  def setUp(self):
    """Set up test fixtures."""
    # Reset the root logger before each test
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
      root_logger.removeHandler(handler)

  def test_configure_logging_root_logger(self):
    """Test configuring the root logger."""
    logger = configure_logging()

    self.assertEqual(logger, logging.getLogger())
    self.assertTrue(logger.handlers)
    self.assertEqual(len(logging.getLogger().handlers), 1)

  def test_configure_logging_named_logger(self):
    """Test configuring a named logger."""
    logger = configure_logging('test_logger')

    self.assertEqual(logger, logging.getLogger('test_logger'))
    self.assertEqual(logger.name, 'test_logger')
    # Named loggers don't have handlers directly, they use the root logger's handlers
    self.assertEqual(len(logging.getLogger().handlers), 1)

  @patch('logging.basicConfig')
  def test_configure_logging_already_configured(self, mock_basic_config):
    """Test that basicConfig is not called if root logger already has handlers."""
    # Add a handler to the root logger
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    root_logger.addHandler(handler)

    logger = configure_logging('test_logger')

    # basicConfig should not be called
    mock_basic_config.assert_not_called()
    self.assertEqual(logger, logging.getLogger('test_logger'))


if __name__ == '__main__':
  unittest.main()
