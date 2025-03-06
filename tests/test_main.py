"""Tests for the main module."""

import os
from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from pdf_to_audiobook import main


@pytest.fixture
def sample_text():
  """Sample extracted text."""
  return 'This is a sample text from a PDF file.'


@pytest.fixture
def sample_audio_data():
  """Sample audio data."""
  return b'sample_audio_data'


@pytest.fixture(autouse=True)
def mock_env_vars():
  """Mock environment variables for testing."""
  with patch.dict(
    os.environ,
    {
      'OPENAI_API_KEY': 'test_openai_key',
      'GEMINI_API_KEY': 'test_gemini_key',
    },
  ):
    yield


def test_main_success():
  """Test the main function with successful conversion."""
  test_args = ['prog', 'test.pdf']

  with (
    patch('argparse.ArgumentParser.parse_args') as mock_args,
    patch('pdf_to_audiobook.main.convert_pdf_to_audiobook', return_value=True),
    patch('os.path.getsize', return_value=1024),
    patch('builtins.print') as mock_print,
  ):
    # Setup mock args
    mock_args.return_value.pdf_path = 'test.pdf'
    mock_args.return_value.output = None
    mock_args.return_value.voice = 'alloy'
    mock_args.return_value.model = 'tts-1'
    mock_args.return_value.speed = 1.0
    mock_args.return_value.min_chunk_size = None

    # Call main function
    with patch('sys.argv', test_args):
      result = main.main()

    # Check results
    assert result == 0
    assert mock_print.call_count >= 2
    assert any(
      'Successfully converted' in call.args[0] for call in mock_print.call_args_list
    )


def test_main_failure():
  """Test the main function with failed conversion."""
  test_args = ['prog', 'test.pdf']

  with (
    patch('argparse.ArgumentParser.parse_args') as mock_args,
    patch('pdf_to_audiobook.main.convert_pdf_to_audiobook', return_value=False),
    patch('builtins.print') as mock_print,
  ):
    # Setup mock args
    mock_args.return_value.pdf_path = 'test.pdf'
    mock_args.return_value.output = None
    mock_args.return_value.voice = 'alloy'
    mock_args.return_value.model = 'tts-1'
    mock_args.return_value.speed = 1.0
    mock_args.return_value.min_chunk_size = None

    # Call main function
    with patch('sys.argv', test_args):
      result = main.main()

    # Check results
    assert result == 1
    assert mock_print.call_count >= 1
    assert any(
      'Failed to convert' in call.args[0] for call in mock_print.call_args_list
    )


def test_main_custom_options():
  """Test the main function with custom options."""
  test_args = [
    'prog',
    'test.pdf',
    '--output',
    'custom_output.mp3',
    '--voice',
    'nova',
    '--model',
    'tts-1-hd',
    '--speed',
    '1.5',
    '--min-chunk-size',
    '800',
  ]

  with (
    patch('argparse.ArgumentParser.parse_args') as mock_args,
    patch(
      'pdf_to_audiobook.main.convert_pdf_to_audiobook', return_value=True
    ) as mock_convert,
    patch('os.path.getsize', return_value=1024),
    patch('builtins.print'),
  ):
    # Setup mock args
    mock_args.return_value.pdf_path = 'test.pdf'
    mock_args.return_value.output = 'custom_output.mp3'
    mock_args.return_value.voice = 'nova'
    mock_args.return_value.model = 'tts-1-hd'
    mock_args.return_value.speed = 1.5
    mock_args.return_value.min_chunk_size = 800

    # Call main function
    with patch('sys.argv', test_args):
      main.main()

    # Check that convert was called with the right args
    mock_convert.assert_called_once_with(
      pdf_path='test.pdf',
      output_path='custom_output.mp3',
      voice='nova',
      model='tts-1-hd',
      speed=1.5,
      min_chunk_size=800,
    )


def test_convert_pdf_to_audiobook_success(sample_text, sample_audio_data):
  """Test successful conversion of PDF to audiobook."""
  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.synthesize_long_text', return_value=sample_audio_data),
    patch('builtins.open', mock_open()) as mock_file,
    patch('os.path.getsize', return_value=1024),
  ):
    # Use a test output path
    test_output_path = 'test_output.mp3'
    result = main.convert_pdf_to_audiobook('test.pdf', output_path=test_output_path)

    assert result is True
    mock_file.assert_called_once_with(test_output_path, 'wb')
    mock_file().write.assert_called_once_with(sample_audio_data)


def test_convert_pdf_to_audiobook_extraction_failed():
  """Test conversion when text extraction fails."""
  with patch('pdf_to_audiobook.main.read_pdf', return_value=None):
    result = main.convert_pdf_to_audiobook('test.pdf')

    assert result is False


def test_convert_pdf_to_audiobook_tts_failed(sample_text):
  """Test conversion when TTS fails."""
  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.synthesize_long_text', return_value=None),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf')

    assert result is False


def test_convert_pdf_to_audiobook_file_save_failed(sample_text, sample_audio_data):
  """Test conversion when file saving fails."""
  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.synthesize_long_text', return_value=sample_audio_data),
    patch('builtins.open', side_effect=Exception('File save error')),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf')

    assert result is False


def test_convert_pdf_to_audiobook_custom_output(sample_text, sample_audio_data):
  """Test conversion with custom output path."""
  custom_output = 'custom_output.mp3'

  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.synthesize_long_text', return_value=sample_audio_data),
    patch('builtins.open', mock_open()) as mock_file,
    patch('os.path.getsize', return_value=1024),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf', output_path=custom_output)

    assert result is True
    mock_file.assert_called_once_with(custom_output, 'wb')


def test_convert_pdf_to_audiobook_custom_voice(sample_text, sample_audio_data):
  """Test conversion with custom voice."""
  custom_voice = 'nova'

  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch(
      'pdf_to_audiobook.main.synthesize_long_text', return_value=sample_audio_data
    ) as mock_tts,
    patch('builtins.open', mock_open()),
    patch('os.path.getsize', return_value=1024),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf', voice=custom_voice)

    assert result is True
    # Check that the voice parameter was passed correctly
    assert mock_tts.call_args.kwargs['voice'] == custom_voice


def test_convert_pdf_to_audiobook_custom_model_and_speed(
  sample_text, sample_audio_data
):
  """Test conversion with custom model and speed."""
  custom_model = 'tts-1-hd'
  custom_speed = 1.5

  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch(
      'pdf_to_audiobook.main.synthesize_long_text', return_value=sample_audio_data
    ) as mock_tts,
    patch('builtins.open', mock_open()),
    patch('os.path.getsize', return_value=1024),
  ):
    result = main.convert_pdf_to_audiobook(
      'test.pdf', model=custom_model, speed=custom_speed
    )

    assert result is True
    # Check that the custom parameters were passed correctly
    assert mock_tts.call_args.kwargs['model'] == custom_model
    assert mock_tts.call_args.kwargs['speed'] == custom_speed


def test_get_default_output_path():
  """Test the get_default_output_path function."""
  pdf_path = 'test.pdf'
  expected_output = os.path.abspath('test.mp3')

  result = main.get_default_output_path(pdf_path)

  assert result == expected_output
