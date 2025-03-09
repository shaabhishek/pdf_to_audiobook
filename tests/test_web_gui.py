"""Tests for the web GUI module.

This module tests the web GUI functionality and integration with the main converter.
"""

import os
import tempfile
import unittest
from unittest import mock

from pdf_to_audiobook.web_gui import get_default_output_folder
from pdf_to_audiobook.web_gui import process_pdf_file
from pdf_to_audiobook.web_gui import update_output_folder


class TestWebGUI(unittest.TestCase):
  """Test cases for the web GUI module."""

  def test_get_default_output_folder(self):
    """Test the get_default_output_folder function."""
    # Test with a typical PDF path
    pdf_path = '/path/to/some/folder/document.pdf'
    expected_folder = '/path/to/some/folder'
    self.assertEqual(get_default_output_folder(pdf_path), expected_folder)

    # Test with empty input
    self.assertEqual(get_default_output_folder(''), '')

  def test_update_output_folder(self):
    """Test the update_output_folder function."""
    # Test with None input
    self.assertEqual(update_output_folder(None), '')

    # Test with a mock file object
    mock_file = mock.MagicMock()
    mock_file.name = '/path/to/test/file.pdf'

    expected_folder = '/path/to/test'
    self.assertEqual(update_output_folder(mock_file), expected_folder)

  @unittest.skip('Integration test that requires Gradio')
  def test_process_pdf_file(self):
    """Test the process_pdf_file function."""
    # Create a temp file to simulate PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
      pdf_path = tmp.name

    try:
      # Create a temporary audio file to simulate the output
      with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
        audio_path = tmp_audio.name

      # Mock the convert_pdf_to_audiobook to avoid actual processing
      with mock.patch(
        'pdf_to_audiobook.web_gui.convert_pdf_to_audiobook', return_value=True
      ):
        # Mock get_output_paths to return a known value
        mock_output_paths = mock.MagicMock()
        mock_output_paths.audio_path = audio_path
        mock_output_paths.text_path = '/path/to/output/file.md'

        with mock.patch(
          'pdf_to_audiobook.web_gui.get_output_paths', return_value=mock_output_paths
        ):
          # Mock the os.path.exists to return True for our audio file
          with mock.patch('os.path.exists', return_value=True):
            # Mock shutil.copy2 to avoid actual copying
            with mock.patch('shutil.copy2') as mock_copy:
              status, result_path = process_pdf_file(
                pdf_file=pdf_path,
                output_folder='/path/to/output',
                custom_title='Test Title',
                voice='alloy',
                tts_model='tts-1',
                speed=1.0,
                min_chunk_size=0,
                ai_model='openai',
              )

              # Verify results
              self.assertIn('Success', status)
              self.assertIn(os.path.basename(audio_path), result_path)
              mock_copy.assert_called_once()
    finally:
      # Clean up temp files
      if os.path.exists(pdf_path):
        os.unlink(pdf_path)
      if os.path.exists(audio_path):
        os.unlink(audio_path)


if __name__ == '__main__':
  unittest.main()
