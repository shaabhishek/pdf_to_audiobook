"""Tests for the tts_client module."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from pdf_to_audiobook import tts_client
from pdf_to_audiobook.config import MAX_TTS_CHUNK_SIZE
from pdf_to_audiobook.tts_client import OpenAITTSEngine
from pdf_to_audiobook.tts_client import TTSClient
from pdf_to_audiobook.tts_client import TTSConfig
from pdf_to_audiobook.tts_client import config


@pytest.fixture
def sample_text():
  """Sample text for TTS conversion."""
  return 'This is a sample text for testing TTS conversion.'


@pytest.fixture
def sample_audio_data():
  """Sample audio data returned by TTS API."""
  return b'sample_audio_data'


@pytest.fixture
def mock_openai_response(sample_audio_data):
  """Mock the OpenAI audio.speech.create response."""
  mock_response = MagicMock()
  mock_response.content = sample_audio_data
  return mock_response


@pytest.fixture
def reset_config():
  """Reset any modified config values after the test."""
  # Store original values
  original_api_key = config.openai_api_key
  original_voice = config.default_voice
  yield
  # Restore original values
  config.openai_api_key = original_api_key
  config.default_voice = original_voice


@pytest.fixture(autouse=True)
def mock_env_vars():
  """Mock environment variables for testing."""
  with patch.dict(
    'os.environ',
    {
      'OPENAI_API_KEY': 'test_openai_key',
    },
  ):
    yield


def test_tts_config_initialization():
  """Test TTSConfig initialization with default values."""
  test_config = TTSConfig()

  # Check basic properties
  assert test_config.default_voice == 'alloy'
  assert test_config.engine_type == 'openai'
  assert len(test_config.section_markers) > 0

  # Test dictionary-like access
  assert test_config['default_voice'] == 'alloy'
  test_config['default_voice'] = 'nova'
  assert test_config.default_voice == 'nova'


def test_openai_tts(mock_openai_response, sample_text, reset_config):
  """Test OpenAI TTS engine."""
  with patch('openai.audio.speech.create', return_value=mock_openai_response):
    result = tts_client.synthesize_speech(sample_text)

  assert result == mock_openai_response.content


def test_synthesize_speech_success(
  mock_openai_response, sample_text, sample_audio_data, reset_config
):
  """Test successfully synthesizing speech."""
  with patch('openai.audio.speech.create', return_value=mock_openai_response):
    result = tts_client.synthesize_speech(sample_text)

  assert result == sample_audio_data


def test_synthesize_speech_empty_text():
  """Test synthesizing speech with empty text."""
  result = tts_client.synthesize_speech('')
  assert result is None


def test_synthesize_speech_no_api_key(reset_config):
  """Test synthesizing speech without an API key."""
  # Set API key to None
  config.openai_api_key = None
  result = tts_client.synthesize_speech('test')
  assert result is None


def test_synthesize_speech_api_error(sample_text, reset_config):
  """Test synthesizing speech when the API request fails."""
  with patch('openai.audio.speech.create', side_effect=Exception('API error')):
    result = tts_client.synthesize_speech(sample_text)

  assert result is None


def test_synthesize_long_text(sample_text, sample_audio_data):
  """Test synthesizing a long text by breaking it into chunks."""
  # Create a text with multiple paragraphs that will be split into chunks
  long_text = (sample_text + '\n\n') * 5

  # Create a very long sentence that will need to be split
  long_sentence = (
    'This is a very long sentence that exceeds the maximum chunk size and needs to be split into multiple chunks. '
    * 50
  )
  long_text += long_sentence

  # Mock asyncio.run for OpenAI path which uses async
  with patch('asyncio.run', return_value=sample_audio_data) as mock_run:
    result = tts_client.synthesize_long_text(long_text, voice='alloy')

    # Verify asyncio.run was called
    assert mock_run.call_count == 1
    assert result == sample_audio_data


def test_text_chunker_split_paragraph():
  """Test TextChunker split_paragraph_if_needed method."""
  chunker = tts_client.TextChunker()

  # Create a paragraph that exceeds the maximum chunk size
  long_paragraph = 'This is a sentence. ' * (MAX_TTS_CHUNK_SIZE // 16 + 1)

  # Split the paragraph
  chunks = chunker.split_paragraph_if_needed(long_paragraph)

  # Should have split the paragraph into multiple chunks
  assert len(chunks) > 1
  # Each chunk should be smaller than the maximum chunk size
  for chunk in chunks:
    assert len(chunk) <= MAX_TTS_CHUNK_SIZE


def test_text_chunker_is_section_marker():
  """Test TextChunker is_section_marker method."""
  chunker = tts_client.TextChunker()

  # Test section markers
  assert chunker.is_section_marker('# Introduction')
  assert chunker.is_section_marker('## Section 1')
  assert chunker.is_section_marker('[pause]')
  assert chunker.is_section_marker('Title: My Document')

  # Test non-section markers
  assert not chunker.is_section_marker('This is a regular paragraph.')
  assert not chunker.is_section_marker('')


def test_text_chunker_optimize_chunks():
  """Test TextChunker optimize_chunks method."""
  chunker = tts_client.TextChunker()

  # Create a text with multiple paragraphs and section markers
  text = (
    '# Introduction\n\nThis is the introduction.\n\n## Section 1\n\nThis is section 1.'
  )

  # Optimize chunks
  chunks = chunker.optimize_chunks(text)

  # Check that we got at least one chunk
  assert len(chunks) >= 1


def test_synthesize_speech_with_custom_parameters(
  mock_openai_response, sample_text, reset_config
):
  """Test synthesizing speech with custom parameters."""
  with patch(
    'openai.audio.speech.create', return_value=mock_openai_response
  ) as mock_create:
    result = tts_client.synthesize_speech(
      sample_text, voice='nova', model='tts-1-hd', output_format='aac', speed=1.5
    )

  mock_create.assert_called_once()
  # Verify custom parameters were passed
  call_kwargs = mock_create.call_args.kwargs
  assert call_kwargs['voice'] == 'nova'
  assert call_kwargs['model'] == 'tts-1-hd'
  assert call_kwargs['response_format'] == 'aac'
  assert call_kwargs['speed'] == 1.5


@pytest.mark.asyncio
async def test_synthesize_speech_async_success(
  mock_openai_response, sample_text, sample_audio_data, reset_config
):
  """Test successfully synthesizing speech asynchronously."""
  # Create a mock AsyncOpenAI client with the proper structure
  mock_client = MagicMock()

  # Use AsyncMock for the create method that will be awaited
  mock_create = AsyncMock()
  mock_create.return_value = mock_openai_response

  # Build the mock structure
  mock_client.audio.speech.create = mock_create

  # Patch the AsyncOpenAI class to return our mock client
  with patch('pdf_to_audiobook.tts_client.AsyncOpenAI', return_value=mock_client):
    # Call via the client directly
    client = TTSClient()
    result = await client.synthesize_speech_async(text=sample_text)

  mock_create.assert_called_once()
  assert result == sample_audio_data


def test_with_retry_decorator():
  """Test that the with_retry decorator properly handles retries."""
  # Mock function that fails first then succeeds
  mock_func = MagicMock(side_effect=[Exception('Test error'), 'success'])

  # Apply decorator
  decorated = tts_client.with_retry(max_retries=2)(mock_func)

  # Call the decorated function
  result = decorated()

  # Should have been called twice (once failing, once succeeding)
  assert mock_func.call_count == 2
  assert result == 'success'


@pytest.mark.asyncio
async def test_with_retry_decorator_async():
  """Test that the with_retry decorator works with async functions."""
  # Mock async function that fails first then succeeds
  mock_func = AsyncMock(side_effect=[Exception('Test error'), 'success'])

  # Apply decorator
  decorated = tts_client.with_retry(max_retries=2, is_async=True)(mock_func)

  # Call the decorated function
  result = await decorated()

  # Should have been called twice (once failing, once succeeding)
  assert mock_func.call_count == 2
  assert result == 'success'


@pytest.mark.asyncio
async def test_synthesize_speech_async_api_error(sample_text, reset_config):
  """Test asynchronous speech synthesis when the API request fails with retry."""
  # Create a mock AsyncOpenAI client
  mock_client = MagicMock()
  mock_create = AsyncMock(side_effect=Exception('API error'))
  mock_client.audio.speech.create = mock_create

  # Patch the AsyncOpenAI class and asyncio.sleep
  with patch('pdf_to_audiobook.tts_client.AsyncOpenAI', return_value=mock_client):
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      # Call via the client directly
      client = TTSClient()
      result = await client.synthesize_speech_async(text=sample_text)

  assert result is None
  assert mock_create.call_count >= 1  # Should be called at least once


@pytest.mark.asyncio
async def test_synthesize_long_text_async(sample_text, sample_audio_data):
  """Test asynchronously synthesizing a long text with parallel processing."""
  # Generate a text that's too large to fit in one chunk
  long_sentence = 'This is a test sentence. ' * 100
  section_markers = ['# Section 1', '# Section 2', '# Section 3', '# Section 4']

  # Create several large paragraphs with section markers
  paragraphs = []
  for marker in section_markers:
    # Make each paragraph just under the max size to ensure they don't get split internally
    paragraphs.append(f'{marker}\n{long_sentence[: MAX_TTS_CHUNK_SIZE // 2]}')

  long_text = '\n\n'.join(paragraphs)

  # Mock TextChunker to return our constructed chunks
  with patch(
    'pdf_to_audiobook.tts_client.TextChunker.optimize_chunks',
    return_value=paragraphs,
  ):
    # Create a mock for the async speech synthesis
    async def mock_speech_async(text, **kwargs):
      await asyncio.sleep(0.01)  # Small delay to simulate async processing
      return sample_audio_data

    # Patch the async function
    with patch(
      'pdf_to_audiobook.tts_client.TTSClient.synthesize_speech_async',
      new_callable=AsyncMock,
      side_effect=mock_speech_async,
    ):
      # Run the function
      client = TTSClient()
      result = await client.synthesize_long_text_async(
        long_text,
        concurrency_limit=2,  # Low concurrency to test parallel processing
      )

  # Result should be non-empty
  assert result is not None
  # Should be properly combined
  assert len(result) > 0


@pytest.mark.asyncio
async def test_synthesize_long_text_async_with_failures():
  """Test async long text synthesis with some chunk failures."""
  long_text = (
    'Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4.\n\nParagraph 5.'
  )

  # Mock optimize_chunks to return each paragraph as a separate chunk
  paragraphs = long_text.split('\n\n')
  with patch(
    'pdf_to_audiobook.tts_client.TextChunker.optimize_chunks',
    return_value=paragraphs,
  ):
    # Create an async mock with alternating success/failure
    mock_synthesize = AsyncMock()
    mock_synthesize.side_effect = [
      b'success',  # First call succeeds
      None,  # Second call fails
      b'success',  # Third call succeeds
      None,  # Fourth call fails
      b'success',  # Fifth call succeeds
    ]

    # Patch the function
    with patch(
      'pdf_to_audiobook.tts_client.TTSClient.synthesize_speech_async', mock_synthesize
    ):
      client = TTSClient()
      result = await client.synthesize_long_text_async(long_text, concurrency_limit=2)

  # Should get combined successful chunks (3 * b'success')
  assert result == b'success' * 3


def test_synchronous_wrapper_calls_async_version():
  """Test that the synchronous wrapper calls the async version."""
  with patch('asyncio.run') as mock_run:
    tts_client.synthesize_long_text('Test text', voice='nova', model='tts-1', speed=1.2)

    # Verify asyncio.run was called with synthesize_long_text_async
    assert mock_run.call_count == 1


def test_openai_tts_engine_prepare_params():
  """Test the OpenAI TTS engine parameter preparation."""
  engine = OpenAITTSEngine()

  # Test with all parameters specified
  params = engine._prepare_params(
    text='Test text', voice='nova', model='tts-1-hd', output_format='mp3', speed=1.5
  )

  assert params['input'] == 'Test text'
  assert params['voice'] == 'nova'
  assert params['model'] == 'tts-1-hd'
  assert params['response_format'] == 'mp3'
  assert params['speed'] == 1.5

  # Test with invalid voice (should use default)
  params = engine._prepare_params(text='Test text', voice='invalid_voice')
  assert params['voice'] == 'alloy'  # Default voice

  # Test with empty text (should return None)
  params = engine._prepare_params(text='')
  assert params is None
