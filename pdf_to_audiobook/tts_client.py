"""Text-to-Speech client module.

This module interfaces with OpenAI TTS engine
to convert text into speech using a clean, modular architecture.
"""

import abc
import asyncio
from dataclasses import dataclass
from dataclasses import field
import functools
import time
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

import openai
from openai import AsyncOpenAI
from openai import OpenAI

from pdf_to_audiobook.logging_config import configure_logging

openai_version = getattr(openai, '__version__', 'unknown')

# Type definitions
T = TypeVar('T')
AsyncFunc = Callable[..., Awaitable[Any]]
SyncFunc = Callable[..., Any]

# Configure logging
logger = configure_logging(__name__)
logger.info(f'Using OpenAI version: {openai_version}')


@dataclass
class TTSConfig:
  """Configuration for TTS services."""

  # Import configuration lazily to avoid circular imports
  from pdf_to_audiobook.config import CONCURRENCY_LIMIT
  from pdf_to_audiobook.config import DEFAULT_OUTPUT_FORMAT
  from pdf_to_audiobook.config import DEFAULT_SPEED
  from pdf_to_audiobook.config import DEFAULT_VOICE
  from pdf_to_audiobook.config import MAX_API_RETRIES
  from pdf_to_audiobook.config import MAX_TTS_CHUNK_SIZE
  from pdf_to_audiobook.config import MIN_EFFICIENT_CHUNK_SIZE
  from pdf_to_audiobook.config import OPENAI_API_KEY
  from pdf_to_audiobook.config import TTS_MODEL

  openai_api_key: str = OPENAI_API_KEY
  default_voice: str = DEFAULT_VOICE
  default_speed: float = DEFAULT_SPEED
  default_output_format: str = DEFAULT_OUTPUT_FORMAT
  tts_model: str = TTS_MODEL
  max_api_retries: int = MAX_API_RETRIES
  max_tts_chunk_size: int = MAX_TTS_CHUNK_SIZE
  min_efficient_chunk_size: int = MIN_EFFICIENT_CHUNK_SIZE
  concurrency_limit: int = CONCURRENCY_LIMIT
  engine_type: str = 'openai'

  # Section markers for text chunking
  section_markers: List[str] = field(
    default_factory=lambda: [
      '[pause]',
      '# ',
      '## ',
      '**',
      'Title:',
      'Authors:',
      'Abstract:',
      'Introduction:',
      'Methodology:',
      'Results:',
      'Conclusion:',
    ]
  )

  def __getitem__(self, key: str) -> Any:
    """Allow dictionary-like access to config properties."""
    return getattr(self, key)

  def __setitem__(self, key: str, value: Any) -> None:
    """Allow dictionary-like setting of config properties."""
    setattr(self, key, value)


# Create a global config instance
config = TTSConfig()


def with_retry(func=None, *, max_retries: Optional[int] = None, is_async: bool = False):
  """Decorator to retry a function call on exception.

  Can be used on both sync and async functions.

  Args:
      func: The function to decorate (when used directly).
      max_retries: Maximum number of retry attempts.
      is_async: Whether the function is asynchronous.

  Returns:
      The decorated function with retry logic.
  """

  def decorator(f):
    retries = max_retries or config.max_api_retries

    @functools.wraps(f)
    async def async_wrapper(*args, **kwargs):
      for attempt in range(retries):
        try:
          return await f(*args, **kwargs)
        except Exception as e:
          if attempt == retries - 1:
            logger.error(f'Failed after {retries} attempts: {e}')
            return None
          wait_time = 2**attempt
          logger.warning(f'Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}')
          await asyncio.sleep(wait_time)
      return None

    @functools.wraps(f)
    def sync_wrapper(*args, **kwargs):
      for attempt in range(retries):
        try:
          return f(*args, **kwargs)
        except Exception as e:
          if attempt == retries - 1:
            logger.error(f'Failed after {retries} attempts: {e}')
            return None
          wait_time = 2**attempt
          logger.warning(f'Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}')
          time.sleep(wait_time)
      return None

    return async_wrapper if is_async else sync_wrapper

  # Handle both @with_retry and @with_retry() forms
  if func is None:
    return decorator
  return decorator(func)


class TextChunker:
  """Service for splitting and optimizing text chunks for TTS processing."""

  def __init__(self, config: TTSConfig = config):
    """Initialize the text chunker with config."""
    self.config = config

  def split_paragraph_if_needed(
    self, paragraph: str, max_size: Optional[int] = None
  ) -> List[str]:
    """Split a paragraph into smaller chunks if it exceeds the maximum size."""
    max_size = max_size or self.config.max_tts_chunk_size

    if len(paragraph) <= max_size:
      return [paragraph]

    logger.info(f'Splitting paragraph of {len(paragraph)} chars (max: {max_size})')
    sentences = paragraph.split('. ')
    chunks = []
    current_chunk = ''

    for i, sentence in enumerate(sentences):
      # Add period except for the last sentence
      sentence_with_period = sentence + '. ' if i < len(sentences) - 1 else sentence

      if len(current_chunk) + len(sentence_with_period) <= max_size:
        current_chunk += sentence_with_period
      else:
        if current_chunk:
          chunks.append(current_chunk)
        current_chunk = sentence_with_period

    if current_chunk:
      chunks.append(current_chunk)

    logger.info(f'Paragraph was split into {len(chunks)} chunks')
    return chunks

  def is_section_marker(self, text: str) -> bool:
    """Check if the text is a section marker or heading that should start a new chunk."""
    text = text.strip()
    if not text:
      return False

    # Check against configured section markers
    for marker in self.config.section_markers:
      if text.startswith(marker):
        return True

    # Heuristics for section detection
    if text.isupper() and len(text) < 50:
      return True
    if ':' in text and len(text) < 50:
      return True

    return False

  def optimize_chunks(
    self, text: str, min_chunk_size: Optional[int] = None
  ) -> List[str]:
    """Split long text into optimized chunks for TTS processing."""
    if not text:
      return []

    max_size = self.config.max_tts_chunk_size
    min_chunk_size = min_chunk_size or self.config.min_efficient_chunk_size

    # Split into paragraphs
    paragraphs = text.split('\n\n')
    logger.info(f'Original text: {len(text)} chars, {len(paragraphs)} paragraphs')

    # First pass: split large paragraphs if needed
    raw_chunks = []
    for paragraph in paragraphs:
      if not paragraph.strip():
        continue

      # Split paragraphs that exceed max size
      if len(paragraph) > max_size:
        paragraph_chunks = self.split_paragraph_if_needed(paragraph, max_size)
        raw_chunks.extend(paragraph_chunks)
      else:
        raw_chunks.append(paragraph)

    # Second pass: optimize chunks to avoid too small chunks
    optimized_chunks = []
    current_chunk = ''

    for chunk in raw_chunks:
      is_section = self.is_section_marker(chunk)

      # Start a new chunk if:
      # 1. Current text is a section marker and we have enough text already, or
      # 2. Adding this chunk would exceed max size
      if (is_section and len(current_chunk) >= min_chunk_size) or (
        len(current_chunk) + len(chunk) + 2 > max_size
      ):
        if current_chunk:
          optimized_chunks.append(current_chunk)
          current_chunk = chunk
      else:
        if current_chunk and chunk:
          current_chunk += '\n\n' + chunk
        else:
          current_chunk = chunk

    # Add the last chunk
    if current_chunk:
      optimized_chunks.append(current_chunk)

    logger.info(
      f'Optimized: {len(raw_chunks)} raw chunks → {len(optimized_chunks)} final chunks'
    )
    return optimized_chunks


class TTSEngine(abc.ABC):
  """Abstract base class for TTS engines."""

  def __init__(self, config: TTSConfig = config):
    """Initialize the TTS engine with config."""
    self.config = config

  @abc.abstractmethod
  def synthesize(self, text: str, **kwargs) -> Optional[bytes]:
    """Convert text to speech."""
    pass

  @abc.abstractmethod
  async def synthesize_async(self, text: str, **kwargs) -> Optional[bytes]:
    """Convert text to speech asynchronously."""
    pass


class OpenAITTSEngine(TTSEngine):
  """OpenAI TTS engine implementation."""

  AVAILABLE_VOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
  DEFAULT_VOICE = 'alloy'

  def __init__(self, config: TTSConfig = config):
    """Initialize the OpenAI TTS engine."""
    super().__init__(config)
    # Don't set a global API key here, create client instances for each request

  def _validate_voice(self, voice: str) -> str:
    """Validate the requested voice and return a compatible voice."""
    if voice in self.AVAILABLE_VOICES:
      return voice
    logger.warning(
      f"Voice '{voice}' not available for OpenAI TTS. Using '{self.DEFAULT_VOICE}' instead."
    )
    return self.DEFAULT_VOICE

  def _prepare_params(
    self,
    text: str,
    voice: Optional[str] = None,
    tts_model: Optional[str] = None,
    output_format: Optional[str] = None,
    speed: Optional[float] = None,
  ) -> Optional[Dict[str, Any]]:
    """Prepare parameters for OpenAI TTS API call."""
    if not text:
      logger.error('No text provided for speech synthesis')
      return None

    if not self.config.openai_api_key:
      logger.error('OpenAI API key not provided')
      return None

    # Use provided values or defaults from config
    voice = self._validate_voice(voice or self.config.default_voice)
    logger.info(
      f'Using model: {tts_model} (original input was: {self.config.tts_model!r})'
    )
    output_format = output_format or self.config.default_output_format
    speed = speed or self.config.default_speed

    return {
      'model': tts_model,
      'voice': voice,
      'input': text,
      'speed': speed,
      'response_format': output_format,
    }

  @with_retry
  def synthesize(self, text: str, **kwargs) -> Optional[bytes]:
    """Convert text to speech using OpenAI TTS API."""
    params = self._prepare_params(
      text,
      voice=kwargs.get('voice'),
      tts_model=kwargs.get('tts_model'),
      output_format=kwargs.get('output_format'),
      speed=kwargs.get('speed'),
    )

    if not params:
      return None

    logger.info(
      f'Converting text to speech using OpenAI TTS with voice: {params["voice"]}, model: {params["tts_model"]}, params: {params}'
    )

    # Create a fresh client for each request with explicit API key
    client = OpenAI(api_key=self.config.openai_api_key)

    try:
      response = client.audio.speech.create(**params)
      return response.content
    except Exception as e:
      logger.error(f'Error in synthesize: {type(e).__name__}: {str(e)}')
      raise

  @with_retry(is_async=True)
  async def synthesize_async(self, text: str, **kwargs) -> Optional[bytes]:
    """Convert text to speech using OpenAI TTS API asynchronously."""
    params = self._prepare_params(
      text,
      voice=kwargs.get('voice'),
      tts_model=kwargs.get('tts_model'),
      output_format=kwargs.get('output_format'),
      speed=kwargs.get('speed'),
    )

    if not params:
      return None

    logger.info(
      f'Converting text to speech asynchronously (OpenAI TTS) with params: {params}'
    )
    # Create a fresh client for each request with explicit API key
    client = AsyncOpenAI(api_key=self.config.openai_api_key)

    try:
      response = await client.audio.speech.create(**params)
      return response.content
    except Exception as e:
      logger.error(f'Error in synthesize_async: {type(e).__name__}: {str(e)}')
      raise


class TTSClient:
  """Client for text-to-speech synthesis."""

  def __init__(self, config: TTSConfig = config):
    """Initialize the TTS client with config and engine.

    Args:
        config: Configuration for the TTS client.
    """
    self.config = config
    logger.info(f'Initializing TTSClient with TTS model: {self.config.tts_model!r}')
    self.text_chunker = TextChunker(config)
    self.engine = OpenAITTSEngine(config)

  def synthesize_speech(self, text: str, **kwargs) -> Optional[bytes]:
    """Convert text to speech using the configured TTS engine."""
    if not text:
      logger.error('No text provided for speech synthesis')
      return None

    return self.engine.synthesize(text=text, **kwargs)

  async def synthesize_speech_async(self, text: str, **kwargs) -> Optional[bytes]:
    """Convert text to speech asynchronously."""
    if not text:
      logger.error('No text provided for speech synthesis')
      return None

    return await self.engine.synthesize_async(text=text, **kwargs)

  async def synthesize_long_text_async(
    self,
    text: str,
    min_chunk_size: Optional[int] = None,
    concurrency_limit: Optional[int] = None,
    **kwargs,
  ) -> Optional[bytes]:
    """Convert long text to speech asynchronously by breaking it into chunks."""
    if not text:
      logger.error('No text provided for speech synthesis')
      return None

    # Split text into optimized chunks
    optimized_chunks = self.text_chunker.optimize_chunks(text, min_chunk_size)
    if not optimized_chunks:
      logger.error('No chunks were generated from the input text')
      return None

    # Limit concurrent API calls
    concurrency_limit = concurrency_limit or self.config.concurrency_limit
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def process_chunk(i, chunk):
      async with semaphore:
        logger.info(
          f'Processing chunk {i + 1}/{len(optimized_chunks)} ({len(chunk)} chars)'
        )
        audio_data = await self.synthesize_speech_async(text=chunk, **kwargs)
        return i, audio_data

    # Process all chunks concurrently with limits
    tasks = [process_chunk(i, chunk) for i, chunk in enumerate(optimized_chunks)]
    results = await asyncio.gather(*tasks)

    # Combine results in correct order
    audio_chunks = [data for _, data in sorted(results) if data is not None]

    if not audio_chunks:
      logger.error('No audio chunks were successfully synthesized')
      return None

    logger.info(f'Combining {len(audio_chunks)} audio chunks')
    return b''.join(audio_chunks)

  def synthesize_long_text(self, text: str, **kwargs) -> Optional[bytes]:
    """Convert long text to speech by breaking it into chunks."""
    return asyncio.run(self.synthesize_long_text_async(text=text, **kwargs))


# Simple interface functions that use the global client
def synthesize_speech(text: str, **kwargs) -> Optional[bytes]:
  """Convert text to speech."""
  # Create a fresh client for each call
  client = TTSClient()
  return client.synthesize_speech(text=text, **kwargs)


async def synthesize_speech_async(text: str, **kwargs) -> Optional[bytes]:
  """Convert text to speech asynchronously."""
  # Create a fresh client for each call
  client = TTSClient()
  return await client.synthesize_speech_async(text=text, **kwargs)


def synthesize_long_text(text: str, **kwargs) -> Optional[bytes]:
  """Convert long text to speech by breaking it into chunks."""
  # Create a fresh client for each call
  client = TTSClient()
  return client.synthesize_long_text(text=text, **kwargs)
