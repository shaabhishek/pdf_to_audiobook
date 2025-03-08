"""Tests for the file_utils module."""

import os
import tempfile
import unittest
from unittest.mock import patch

from pdf_to_audiobook.file_utils import ensure_directory_exists
from pdf_to_audiobook.file_utils import save_binary_to_file
from pdf_to_audiobook.file_utils import save_text_to_file


class TestFileUtils(unittest.TestCase):
  """Test cases for the file_utils module."""

  def test_ensure_directory_exists_success(self):
    """Test ensuring a directory exists when it doesn't yet."""
    with tempfile.TemporaryDirectory() as temp_dir:
      test_dir = os.path.join(temp_dir, 'test_dir')
      self.assertFalse(os.path.exists(test_dir))

      result = ensure_directory_exists(test_dir)

      self.assertTrue(result)
      self.assertTrue(os.path.exists(test_dir))
      self.assertTrue(os.path.isdir(test_dir))

  def test_ensure_directory_exists_already_exists(self):
    """Test ensuring a directory exists when it already does."""
    with tempfile.TemporaryDirectory() as temp_dir:
      self.assertTrue(os.path.exists(temp_dir))

      result = ensure_directory_exists(temp_dir)

      self.assertTrue(result)
      self.assertTrue(os.path.exists(temp_dir))
      self.assertTrue(os.path.isdir(temp_dir))

  @patch('os.makedirs')
  def test_ensure_directory_exists_error(self, mock_makedirs):
    """Test ensuring a directory exists when an error occurs."""
    mock_makedirs.side_effect = PermissionError('Permission denied')

    result = ensure_directory_exists('/test/dir')

    self.assertFalse(result)
    mock_makedirs.assert_called_once_with('/test/dir', exist_ok=True)

  def test_save_text_to_file_success(self):
    """Test saving text to a file successfully."""
    with tempfile.TemporaryDirectory() as temp_dir:
      test_file = os.path.join(temp_dir, 'test.txt')
      test_content = 'This is a test content.'

      result = save_text_to_file(test_content, test_file)

      self.assertIsNotNone(result)
      self.assertEqual(os.path.abspath(test_file), result)
      self.assertTrue(os.path.exists(test_file))

      with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
      self.assertEqual(test_content, content)

  def test_save_text_to_file_with_subdirectory(self):
    """Test saving text to a file in a subdirectory that doesn't exist yet."""
    with tempfile.TemporaryDirectory() as temp_dir:
      test_subdir = os.path.join(temp_dir, 'subdir')
      test_file = os.path.join(test_subdir, 'test.txt')
      test_content = 'This is a test content.'

      result = save_text_to_file(test_content, test_file)

      self.assertIsNotNone(result)
      self.assertEqual(os.path.abspath(test_file), result)
      self.assertTrue(os.path.exists(test_file))
      self.assertTrue(os.path.isdir(test_subdir))

      with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
      self.assertEqual(test_content, content)

  @patch('pdf_to_audiobook.file_utils.ensure_directory_exists')
  def test_save_text_to_file_directory_error(self, mock_ensure_dir):
    """Test saving text to a file when directory creation fails."""
    mock_ensure_dir.return_value = False

    result = save_text_to_file('test content', '/test/file.txt')

    self.assertIsNone(result)
    mock_ensure_dir.assert_called_once()

  def test_save_binary_to_file_success(self):
    """Test saving binary data to a file successfully."""
    with tempfile.TemporaryDirectory() as temp_dir:
      test_file = os.path.join(temp_dir, 'test.bin')
      test_content = b'This is binary content.'

      result = save_binary_to_file(test_content, test_file)

      self.assertIsNotNone(result)
      self.assertEqual(os.path.abspath(test_file), result)
      self.assertTrue(os.path.exists(test_file))

      with open(test_file, 'rb') as f:
        content = f.read()
      self.assertEqual(test_content, content)

  @patch('pdf_to_audiobook.file_utils.ensure_directory_exists')
  def test_save_binary_to_file_directory_error(self, mock_ensure_dir):
    """Test saving binary data to a file when directory creation fails."""
    mock_ensure_dir.return_value = False

    result = save_binary_to_file(b'test content', '/test/file.bin')

    self.assertIsNone(result)
    mock_ensure_dir.assert_called_once()


if __name__ == '__main__':
  unittest.main()
