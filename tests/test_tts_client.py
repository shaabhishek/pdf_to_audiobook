"""Tests for the tts_client module."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from pdf_to_audiobook import tts_client
from pdf_to_audiobook.config import MAX_TTS_CHUNK_SIZE


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


def test_synthesize_speech_success(
  mock_openai_response, sample_text, sample_audio_data
):
  """Test successfully synthesizing speech."""
  with patch('openai.audio.speech.create', return_value=mock_openai_response):
    result = tts_client.synthesize_speech(sample_text)

  assert result == sample_audio_data


def test_synthesize_speech_empty_text():
  """Test synthesizing speech with empty text."""
  result = tts_client.synthesize_speech('')

  assert result is None


def test_synthesize_speech_no_api_key():
  """Test synthesizing speech without an API key."""
  with patch('pdf_to_audiobook.tts_client.OPENAI_API_KEY', None):
    result = tts_client.synthesize_speech('test')

  assert result is None


def test_synthesize_speech_api_error(sample_text):
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

  # In the new implementation, synthesize_long_text calls synthesize_long_text_async
  # which then calls synthesize_speech_async for each chunk
  with patch(
    'asyncio.run',
    return_value=sample_audio_data * 5,  # Simulate 5 chunks of audio
  ) as mock_run:
    result = tts_client.synthesize_long_text(long_text)

  assert mock_run.call_count == 1
  assert result == sample_audio_data * 5


def test_split_paragraph_if_needed():
  """Test splitting a paragraph if it exceeds the maximum chunk size."""
  # Create a paragraph that exceeds the maximum chunk size
  long_paragraph = 'This is a sentence. ' * (MAX_TTS_CHUNK_SIZE // 16 + 1)

  # Split the paragraph
  chunks = tts_client.split_paragraph_if_needed(long_paragraph)

  # Should have split the paragraph into multiple chunks
  assert len(chunks) > 1
  # Each chunk should be smaller than the maximum chunk size
  for chunk in chunks:
    assert len(chunk) <= MAX_TTS_CHUNK_SIZE


def test_synthesize_long_text_empty():
  """Test synthesizing an empty long text."""
  result = tts_client.synthesize_long_text('')

  assert result is None


def test_synthesize_long_text_all_chunks_fail():
  """Test synthesizing a long text when all chunks fail."""
  long_text = 'This is a sample text for testing.\n\nIt has multiple paragraphs.' * 10

  # Mock asyncio.run to return None, simulating that no audio chunks were synthesized
  with patch('asyncio.run', return_value=None):
    result = tts_client.synthesize_long_text(long_text)

  assert result is None


def test_synthesize_speech_with_custom_parameters(mock_openai_response, sample_text):
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
  mock_openai_response, sample_text, sample_audio_data
):
  """Test successfully synthesizing speech asynchronously."""
  # Create a mock AsyncOpenAI client with the proper structure
  mock_client = MagicMock()
  mock_speech = MagicMock()

  # Use AsyncMock for the create method that will be awaited
  mock_create = AsyncMock()
  mock_create.return_value = mock_openai_response

  # Build the mock structure
  mock_client.audio.speech.create = mock_create

  # Patch the AsyncOpenAI class to return our mock client
  with patch('pdf_to_audiobook.tts_client.AsyncOpenAI', return_value=mock_client):
    result = await tts_client.synthesize_speech_async(sample_text)

  mock_create.assert_called_once()
  assert result == sample_audio_data


@pytest.mark.asyncio
async def test_synthesize_speech_async_api_error(sample_text):
  """Test asynchronous speech synthesis when the API request fails with retry."""
  # Create a mock AsyncOpenAI client with the proper structure
  mock_client = MagicMock()

  # Use AsyncMock for the create method that will be awaited
  mock_create = AsyncMock(side_effect=Exception('API error'))

  # Build the mock structure
  mock_client.audio.speech.create = mock_create

  # Patch the AsyncOpenAI class and asyncio.sleep
  with patch('pdf_to_audiobook.tts_client.AsyncOpenAI', return_value=mock_client):
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await tts_client.synthesize_speech_async(sample_text, max_retries=2)

  assert result is None
  assert mock_create.call_count == 2  # Should be called twice (initial + 1 retry)
  assert mock_sleep.call_count == 1  # Should sleep once between retries


@pytest.mark.asyncio
async def test_synthesize_long_text_async(sample_text, sample_audio_data):
  """Test asynchronously synthesizing a long text with parallel processing."""
  # Create text that will definitely be split into multiple chunks
  # We'll use the MAX_TTS_CHUNK_SIZE to force splitting

  # Generate a text that's too large to fit in one chunk
  long_sentence = 'This is a test sentence. ' * 100
  section_markers = ['# Section 1', '# Section 2', '# Section 3', '# Section 4']

  # Create several large paragraphs with section markers
  paragraphs = []
  for marker in section_markers:
    # Make each paragraph just under the max size to ensure they don't get split internally
    paragraphs.append(f'{marker}\n{long_sentence[: MAX_TTS_CHUNK_SIZE // 2]}')

  long_text = '\n\n'.join(paragraphs)

  # Mock split_paragraph_if_needed to return chunks as-is (no further splitting)
  with patch(
    'pdf_to_audiobook.tts_client.split_paragraph_if_needed',
    side_effect=lambda p, _: [p],
  ):
    # Create a mock for the async speech synthesis
    async def mock_speech_async(text, **kwargs):
      await asyncio.sleep(0.01)  # Small delay to simulate async processing
      return sample_audio_data

    mock_synthesize = AsyncMock(side_effect=mock_speech_async)

    # Patch the async function
    with patch('pdf_to_audiobook.tts_client.synthesize_speech_async', mock_synthesize):
      # Run the function
      result = await tts_client.synthesize_long_text_async(
        long_text,
        concurrency_limit=2,  # Low concurrency to test parallel processing
      )

  # Should have called synthesize_speech_async at least once per section
  assert mock_synthesize.call_count >= len(section_markers)

  # Result should be the combined audio data
  assert result == sample_audio_data * mock_synthesize.call_count


@pytest.mark.asyncio
async def test_synthesize_long_text_async_with_failures():
  """Test async long text synthesis with some chunk failures."""
  long_text = (
    'Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4.\n\nParagraph 5.'
  )

  # Mock to alternate between success and failure
  call_count = 0

  async def mock_with_alternating_results(text, **kwargs):
    nonlocal call_count
    call_count += 1
    # Every other call fails
    if call_count % 2 == 0:
      return None
    return b'success'

  # Create an async mock with our side effect
  mock_synthesize = AsyncMock(side_effect=mock_with_alternating_results)

  # Patch the function and ensure we have multiple chunks
  with patch('pdf_to_audiobook.tts_client.synthesize_speech_async', mock_synthesize):
    with patch('pdf_to_audiobook.tts_client.is_section_marker', return_value=True):
      result = await tts_client.synthesize_long_text_async(
        long_text, concurrency_limit=2
      )

  # Should still get a result with the successful chunks
  assert result == b'success' * ((call_count + 1) // 2)


def test_synchronous_wrapper_calls_async_version():
  """Test that the synchronous wrapper calls the async version."""
  with patch('asyncio.run') as mock_run:
    tts_client.synthesize_long_text('Test text', voice='nova', model='tts-1', speed=1.2)

    # Verify asyncio.run was called with synthesize_long_text_async
    assert mock_run.call_count == 1
    # Get the coroutine object that was passed to asyncio.run
    coro = mock_run.call_args[0][0]
    # Check that it's calling our expected function with the right parameters
    assert coro.cr_code.co_name == 'synthesize_long_text_async'
