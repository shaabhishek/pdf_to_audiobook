"""Tests for the main module."""

import os
import pathlib
from unittest.mock import patch

import pytest

from pdf_to_audiobook import main
from pdf_to_audiobook.main import OutputPaths


@pytest.fixture
def sample_text():
  """Sample extracted text."""
  return 'This is a sample text from a PDF file.'


@pytest.fixture
def sample_audio_data():
  """Sample audio data."""
  return b'sample_audio_data'


@pytest.fixture
def output_paths():
  """Common OutputPaths instance for testing."""
  return OutputPaths(
    text_path=pathlib.Path('text_output.md'),
    audio_path=pathlib.Path('test_gemini.mp3'),
    title='test_title',
  )


@pytest.fixture
def custom_output_paths():
  """OutputPaths with custom output folder."""
  custom_output_folder = 'custom_output_folder'
  return OutputPaths(
    text_path=pathlib.Path(f'{custom_output_folder}/test_title_gemini.md'),
    audio_path=pathlib.Path(f'{custom_output_folder}/test_title_gemini.mp3'),
    title='test_title',
  )


@pytest.fixture
def mock_path_stat():
  """Mock for Path.stat method that returns file size of 1024 bytes."""
  return type('obj', (object,), {'st_size': 1024})


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


# Main function tests
def test_main_success(output_paths, mock_path_stat):
  """Test the main function with successful conversion."""
  with (
    patch('argparse.ArgumentParser.parse_args') as mock_args,
    patch('pdf_to_audiobook.main.convert_pdf_to_audiobook', return_value=True),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=output_paths),
    patch('pathlib.Path.stat', return_value=mock_path_stat),
    patch('builtins.print') as mock_print,
  ):
    # Setup mock args
    mock_args.return_value.pdf_path = 'test.pdf'
    mock_args.return_value.output = None
    mock_args.return_value.voice = 'alloy'
    mock_args.return_value.tts_model = 'tts-1'
    mock_args.return_value.speed = 1.0
    mock_args.return_value.min_chunk_size = None
    mock_args.return_value.ai_model = 'gemini'
    mock_args.return_value.title = None

    # Call main
    result = main.main()

    # Check that convert_pdf_to_audiobook was called with the correct arguments
    main.convert_pdf_to_audiobook.assert_called_once_with(
      pdf_path='test.pdf',
      output_folder=None,
      voice='alloy',
      tts_model='tts-1',
      speed=1.0,
      min_chunk_size=None,
      ai_model='gemini',
      custom_title=None,
    )

    # Check that the function returned success
    assert result == 0

    # Check that the success message was printed
    mock_print.assert_any_call('✅ Successfully converted test.pdf to audiobook:')


def test_main_failure():
  """Test the main function with failed conversion."""
  with (
    patch('argparse.ArgumentParser.parse_args') as mock_args,
    patch('pdf_to_audiobook.main.convert_pdf_to_audiobook', return_value=False),
    patch('builtins.print') as mock_print,
  ):
    # Setup mock args
    mock_args.return_value.pdf_path = 'test.pdf'
    mock_args.return_value.output = None
    mock_args.return_value.voice = 'alloy'
    mock_args.return_value.tts_model = 'tts-1'
    mock_args.return_value.speed = 1.0
    mock_args.return_value.min_chunk_size = None
    mock_args.return_value.ai_model = 'gemini'
    mock_args.return_value.title = None

    # Call main
    result = main.main()

    # Check that the function returned failure
    assert result == 1

    # Check that the failure message was printed
    mock_print.assert_called_once_with(
      '❌ Failed to convert PDF to audiobook. Check logs for details.'
    )


def test_main_custom_options(mock_path_stat):
  """Test the main function with custom options."""
  custom_paths = OutputPaths(
    text_path=pathlib.Path('custom_output_folder/custom_title_openai.md'),
    audio_path=pathlib.Path('custom_output_folder/custom_title_openai.mp3'),
    title='custom_title',
  )

  with (
    patch('argparse.ArgumentParser.parse_args') as mock_args,
    patch('pdf_to_audiobook.main.convert_pdf_to_audiobook', return_value=True),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=custom_paths),
    patch('pathlib.Path.stat', return_value=mock_path_stat),
    patch('builtins.print') as mock_print,
  ):
    # Setup mock args
    mock_args.return_value.pdf_path = 'test.pdf'
    mock_args.return_value.output = 'custom_output_folder'
    mock_args.return_value.voice = 'nova'
    mock_args.return_value.tts_model = 'tts-1-hd'
    mock_args.return_value.speed = 1.5
    mock_args.return_value.min_chunk_size = 1000
    mock_args.return_value.ai_model = 'openai'
    mock_args.return_value.title = 'Custom Paper Title'

    # Call main
    result = main.main()

    # Check that convert_pdf_to_audiobook was called with the correct arguments
    main.convert_pdf_to_audiobook.assert_called_once_with(
      pdf_path='test.pdf',
      output_folder='custom_output_folder',
      voice='nova',
      tts_model='tts-1-hd',
      speed=1.5,
      min_chunk_size=1000,
      ai_model='openai',
      custom_title='Custom Paper Title',
    )

    assert result == 0
    mock_print.assert_any_call('✅ Successfully converted test.pdf to audiobook:')


# Test PDF to audiobook conversion functions
def test_convert_pdf_to_audiobook_success(sample_text, sample_audio_data, output_paths):
  """Test the convert_pdf_to_audiobook function with successful conversion."""
  with (
    patch('pdf_to_audiobook.main.process_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.convert_text_to_audio', return_value=True),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=output_paths),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf')

    assert result is True
    main.process_pdf.assert_called_once_with('test.pdf', output_paths, 'gemini')
    main.convert_text_to_audio.assert_called_once_with(
      sample_text,
      output_paths.audio_path,
      voice='alloy',
      tts_model='tts-1  # tts-1, tts-1-hd',
      speed=1.25,
      min_chunk_size=None,
    )


def test_convert_pdf_to_audiobook_extraction_failed(output_paths):
  """Test the convert_pdf_to_audiobook function when text extraction fails."""
  with (
    patch('pdf_to_audiobook.main.process_pdf', return_value=None),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=output_paths),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf')

    assert result is False
    main.process_pdf.assert_called_once_with('test.pdf', output_paths, 'gemini')


def test_convert_pdf_to_audiobook_tts_failed(sample_text, output_paths):
  """Test the convert_pdf_to_audiobook function when TTS conversion fails."""
  with (
    patch('pdf_to_audiobook.main.process_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.convert_text_to_audio', return_value=False),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=output_paths),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf')

    assert result is False
    main.process_pdf.assert_called_once_with('test.pdf', output_paths, 'gemini')
    main.convert_text_to_audio.assert_called_once()


def test_convert_pdf_to_audiobook_custom_output(
  sample_text, sample_audio_data, custom_output_paths
):
  """Test the convert_pdf_to_audiobook function with a custom output folder."""
  custom_output_folder = 'custom_output_folder'

  with (
    patch('pdf_to_audiobook.main.process_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.convert_text_to_audio', return_value=True),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=custom_output_paths),
    patch('pdf_to_audiobook.main.ensure_directory_exists', return_value=True),
  ):
    result = main.convert_pdf_to_audiobook(
      'test.pdf', output_folder=custom_output_folder
    )

    assert result is True
    main.process_pdf.assert_called_once_with('test.pdf', custom_output_paths, 'gemini')
    main.convert_text_to_audio.assert_called_once_with(
      sample_text,
      custom_output_paths.audio_path,
      voice='alloy',
      tts_model='tts-1  # tts-1, tts-1-hd',
      speed=1.25,
      min_chunk_size=None,
    )


def test_convert_pdf_to_audiobook_custom_voice(
  sample_text, sample_audio_data, output_paths
):
  """Test the convert_pdf_to_audiobook function with a custom voice."""
  custom_voice = 'nova'

  with (
    patch('pdf_to_audiobook.main.process_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.convert_text_to_audio', return_value=True),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=output_paths),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf', voice=custom_voice)

    assert result is True
    main.process_pdf.assert_called_once_with('test.pdf', output_paths, 'gemini')
    main.convert_text_to_audio.assert_called_once_with(
      sample_text,
      output_paths.audio_path,
      voice=custom_voice,
      tts_model='tts-1  # tts-1, tts-1-hd',
      speed=1.25,
      min_chunk_size=None,
    )


def test_convert_pdf_to_audiobook_custom_model_and_speed(
  sample_text, sample_audio_data, output_paths
):
  """Test the convert_pdf_to_audiobook function with custom model and speed."""
  custom_model = 'tts-1-hd'
  custom_speed = 1.5

  with (
    patch('pdf_to_audiobook.main.process_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.convert_text_to_audio', return_value=True),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=output_paths),
  ):
    result = main.convert_pdf_to_audiobook(
      'test.pdf', tts_model=custom_model, speed=custom_speed
    )

    assert result is True
    main.process_pdf.assert_called_once_with('test.pdf', output_paths, 'gemini')
    main.convert_text_to_audio.assert_called_once_with(
      sample_text,
      output_paths.audio_path,
      voice='alloy',
      tts_model=custom_model,
      speed=custom_speed,
      min_chunk_size=None,
    )


def test_convert_pdf_to_audiobook_with_custom_title(sample_text, sample_audio_data):
  """Test that the custom title is used for the markdown filename when provided."""
  custom_title = 'Custom Paper Title'
  custom_paths = OutputPaths(
    text_path=pathlib.Path('custom_title_gemini.md'),
    audio_path=pathlib.Path('custom_title_gemini.mp3'),
    title='custom_title',
  )

  with (
    patch('pdf_to_audiobook.main.process_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.convert_text_to_audio', return_value=True),
    patch('pdf_to_audiobook.main.get_output_paths', return_value=custom_paths),
  ):
    result = main.convert_pdf_to_audiobook('test.pdf', custom_title=custom_title)

    assert result is True
    main.get_output_paths.assert_called_once_with(
      'test.pdf', 'gemini', None, custom_title
    )
    main.process_pdf.assert_called_once_with('test.pdf', custom_paths, 'gemini')
    main.convert_text_to_audio.assert_called_once_with(
      sample_text,
      custom_paths.audio_path,
      voice='alloy',
      tts_model='tts-1  # tts-1, tts-1-hd',
      speed=1.25,
      min_chunk_size=None,
    )


# Test text-to-audio conversion functions
def test_convert_text_to_audio_success(sample_text, sample_audio_data, mock_path_stat):
  """Test the convert_text_to_audio function with successful conversion."""
  output_path = pathlib.Path('output.mp3')

  with (
    patch('pdf_to_audiobook.main.synthesize_long_text', return_value=sample_audio_data),
    patch(
      'pdf_to_audiobook.main.save_binary_to_file',
      return_value='/absolute/path/to/output.mp3',
    ),
    patch('pathlib.Path.stat', return_value=mock_path_stat),
  ):
    result = main.convert_text_to_audio(sample_text, output_path)

    assert result is True
    main.synthesize_long_text.assert_called_once_with(
      sample_text,
      voice='alloy',
      tts_model='tts-1  # tts-1, tts-1-hd',
      speed=1.25,
      min_chunk_size=None,
    )


def test_convert_text_to_audio_synthesis_failed(sample_text):
  """Test the convert_text_to_audio function when synthesis fails."""
  output_path = pathlib.Path('output.mp3')

  with patch('pdf_to_audiobook.main.synthesize_long_text', return_value=None):
    result = main.convert_text_to_audio(sample_text, output_path)

    assert result is False
    main.synthesize_long_text.assert_called_once()


def test_convert_text_to_audio_save_failed(sample_text, sample_audio_data):
  """Test the convert_text_to_audio function when saving fails."""
  output_path = pathlib.Path('output.mp3')

  with (
    patch('pdf_to_audiobook.main.synthesize_long_text', return_value=sample_audio_data),
    patch('pdf_to_audiobook.main.save_binary_to_file', return_value=None),
  ):
    result = main.convert_text_to_audio(sample_text, output_path)

    assert result is False
    main.synthesize_long_text.assert_called_once()
    main.save_binary_to_file.assert_called_once_with(
      sample_audio_data, str(output_path)
    )


# Test PDF processing functions
def test_process_pdf_success(sample_text, output_paths):
  """Test process_pdf with successful extraction and saving."""
  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch(
      'pdf_to_audiobook.main.save_text_to_file',
      return_value='/absolute/path/to/text_output.md',
    ),
  ):
    result = main.process_pdf('test.pdf', output_paths, 'gemini')

    assert result == sample_text
    main.read_pdf.assert_called_once_with('test.pdf', ai_model='gemini')
    main.save_text_to_file.assert_called_once_with(
      sample_text, str(output_paths.text_path)
    )


def test_process_pdf_extraction_failed(output_paths):
  """Test process_pdf when text extraction fails."""
  with patch('pdf_to_audiobook.main.read_pdf', return_value=None):
    result = main.process_pdf('test.pdf', output_paths, 'gemini')

    assert result is None
    main.read_pdf.assert_called_once_with('test.pdf', ai_model='gemini')


def test_process_pdf_save_failed(sample_text, output_paths):
  """Test process_pdf when saving the text fails."""
  with (
    patch('pdf_to_audiobook.main.read_pdf', return_value=sample_text),
    patch('pdf_to_audiobook.main.save_text_to_file', return_value=None),
  ):
    result = main.process_pdf('test.pdf', output_paths, 'gemini')

    assert result == sample_text  # Should still return the text even if saving fails
    main.read_pdf.assert_called_once_with('test.pdf', ai_model='gemini')
    main.save_text_to_file.assert_called_once_with(
      sample_text, str(output_paths.text_path)
    )


# Test path handling functions
def test_get_output_paths():
  """Test the get_output_paths function."""
  # Test with relative path
  with patch('pathlib.Path.resolve', return_value=pathlib.Path('/abs/path/test.pdf')):
    with patch(
      'pdf_to_audiobook.main.get_snake_case_title_from_pdf', return_value='test_title'
    ):
      result = main.get_output_paths('test.pdf', 'gemini')

      assert isinstance(result, OutputPaths)
      assert result.title == 'test_title'
      assert result.text_path.name == 'test_title_gemini.md'
      assert result.audio_path.name == 'test_title_gemini.mp3'

  # Test with absolute path
  with patch('pathlib.Path.resolve', return_value=pathlib.Path('/abs/path/test.pdf')):
    with patch(
      'pdf_to_audiobook.main.get_snake_case_title_from_pdf', return_value='test_title'
    ):
      result = main.get_output_paths('/abs/path/test.pdf', 'openai')

      assert isinstance(result, OutputPaths)
      assert result.title == 'test_title'
      assert result.text_path.name == 'test_title_openai.md'
      assert result.audio_path.name == 'test_title_openai.mp3'

  # Test with custom output folder
  with patch('pathlib.Path.resolve', return_value=pathlib.Path('/abs/path/test.pdf')):
    with patch(
      'pdf_to_audiobook.main.get_snake_case_title_from_pdf', return_value='test_title'
    ):
      with patch('pdf_to_audiobook.main.ensure_directory_exists', return_value=True):
        result = main.get_output_paths(
          'test.pdf', 'gemini', output_folder='custom_folder'
        )

        assert isinstance(result, OutputPaths)
        assert result.title == 'test_title'
        assert result.text_path.name == 'test_title_gemini.md'
        assert result.audio_path.name == 'test_title_gemini.mp3'
        assert 'custom_folder' in str(result.text_path)
        assert 'custom_folder' in str(result.audio_path)


def test_get_output_paths_with_custom_title():
  """Test the get_output_paths function with a custom title."""
  with patch('pdf_to_audiobook.main.to_snake_case', return_value='custom_title'):
    result = main.get_output_paths('test.pdf', 'gemini', custom_title='Custom Title')

    assert result.title == 'custom_title'
    assert result.text_path.name == 'custom_title_gemini.md'
    assert result.audio_path.name == 'custom_title_gemini.mp3'
