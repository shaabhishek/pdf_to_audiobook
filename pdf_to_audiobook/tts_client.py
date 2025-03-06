"""Text-to-Speech client module.

This module interfaces with OpenAI's Text-to-Speech API
to convert text into speech.
"""

import asyncio
import logging
import time
from typing import List
from typing import Optional

import openai
from openai import AsyncOpenAI

from pdf_to_audiobook.config import CONCURRENCY_LIMIT
from pdf_to_audiobook.config import DEFAULT_OUTPUT_FORMAT
from pdf_to_audiobook.config import DEFAULT_SPEED
from pdf_to_audiobook.config import DEFAULT_VOICE
from pdf_to_audiobook.config import LOG_LEVEL
from pdf_to_audiobook.config import MAX_API_RETRIES
from pdf_to_audiobook.config import MAX_TTS_CHUNK_SIZE
from pdf_to_audiobook.config import MIN_EFFICIENT_CHUNK_SIZE
from pdf_to_audiobook.config import OPENAI_API_KEY
from pdf_to_audiobook.config import TTS_MODEL

# Configure logging
logging.basicConfig(
  level=getattr(logging, LOG_LEVEL),
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY


def split_paragraph_if_needed(
  paragraph: str, max_size: int = MAX_TTS_CHUNK_SIZE
) -> List[str]:
  """Split a paragraph into smaller chunks if it exceeds the maximum size.

  Args:
      paragraph: The paragraph to split.
      max_size: Maximum size of each chunk.

  Returns:
      List of paragraph chunks.
  """
  if len(paragraph) <= max_size:
    return [paragraph]

  logger.info(f'Splitting paragraph of {len(paragraph)} chars (max: {max_size})')
  logger.info(f'Paragraph begins with: {paragraph[:100]}...')

  # Split by sentences if possible
  sentences = paragraph.split('. ')
  logger.info(f'Split into {len(sentences)} sentences')

  chunks = []
  current_chunk = ''

  for i, sentence in enumerate(sentences):
    sentence_with_period = sentence + '. ' if i < len(sentences) - 1 else sentence
    if len(current_chunk) + len(sentence_with_period) <= max_size:
      current_chunk += sentence_with_period
    else:
      if current_chunk:
        chunks.append(current_chunk)
        logger.info(f'Created chunk of {len(current_chunk)} chars')
      current_chunk = sentence_with_period

  if current_chunk:
    chunks.append(current_chunk)
    logger.info(f'Created final chunk of {len(current_chunk)} chars')

  logger.info(f'Paragraph was split into {len(chunks)} chunks')
  return chunks


def synthesize_speech_with_retry(
  text: str,
  voice: str = DEFAULT_VOICE,
  model: str = TTS_MODEL,
  output_format: str = DEFAULT_OUTPUT_FORMAT,
  speed: float = DEFAULT_SPEED,
  max_retries: int = MAX_API_RETRIES,
) -> Optional[bytes]:
  """Convert text to speech using OpenAI TTS API with retry logic.

  Args:
      text: The text to convert to speech.
      voice: The voice to use for synthesis (alloy, echo, fable, onyx, nova, shimmer).
      model: TTS model to use (e.g., tts-1).
      output_format: The audio format for the output (mp3, opus, aac, flac).
      speed: Speech speed multiplier (0.25 to 4.0).
      max_retries: Maximum number of retry attempts.

  Returns:
      Audio data in bytes or None if synthesis failed.
  """
  if not text:
    logger.error('No text provided for speech synthesis')
    return None

  if not OPENAI_API_KEY:
    logger.error('OpenAI API key not provided')
    return None

  logger.info(f'Converting text to speech using voice: {voice}, model: {model}')

  for attempt in range(max_retries):
    try:
      # Call OpenAI TTS API
      response = openai.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        speed=speed,
        response_format=output_format,
      )

      # Get audio content as bytes
      audio_data = response.content

      logger.info('Successfully converted text to speech')
      return audio_data

    except Exception as e:
      if attempt == max_retries - 1:
        logger.error(f'Failed after {max_retries} attempts: {e}')
        return None

      wait_time = 2**attempt
      logger.warning(f'Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}')
      time.sleep(wait_time)


def synthesize_speech(
  text: str,
  voice: str = DEFAULT_VOICE,
  model: str = TTS_MODEL,
  output_format: str = DEFAULT_OUTPUT_FORMAT,
  speed: float = DEFAULT_SPEED,
) -> Optional[bytes]:
  """Convert text to speech using OpenAI TTS API.

  Args:
      text: The text to convert to speech.
      voice: The voice to use for synthesis (alloy, echo, fable, onyx, nova, shimmer).
      model: TTS model to use (e.g., tts-1).
      output_format: The audio format for the output (mp3, opus, aac, flac).
      speed: Speech speed multiplier (0.25 to 4.0).

  Returns:
      Audio data in bytes or None if synthesis failed.
  """
  return synthesize_speech_with_retry(text, voice, model, output_format, speed)


def is_section_marker(text: str) -> bool:
  """Check if the text is a section marker or heading that should start a new chunk.

  Args:
      text: The text to check.

  Returns:
      True if the text is a section marker or heading, False otherwise.
  """
  section_break_markers = [
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
  text = text.strip()

  if not text:
    return False

  # Check if text starts with any of the section markers
  for marker in section_break_markers:
    if text.startswith(marker):
      return True

  # Check if text is all caps and relatively short (likely a heading)
  if text.isupper() and len(text) < 50:
    return True

  # Check if the text contains ":" and is relatively short (likely a heading)
  if ':' in text and len(text) < 50:
    return True

  return False


async def synthesize_speech_async(
  text: str,
  voice: str = DEFAULT_VOICE,
  model: str = TTS_MODEL,
  output_format: str = DEFAULT_OUTPUT_FORMAT,
  speed: float = DEFAULT_SPEED,
  max_retries: int = MAX_API_RETRIES,
) -> Optional[bytes]:
  """Convert text to speech using OpenAI TTS API asynchronously with retry logic.

  Args:
      text: The text to convert to speech.
      voice: The voice to use for synthesis (alloy, echo, fable, onyx, nova, shimmer).
      model: TTS model to use (e.g., tts-1).
      output_format: The audio format for the output (mp3, opus, aac, flac).
      speed: Speech speed multiplier (0.25 to 4.0).
      max_retries: Maximum number of retry attempts.

  Returns:
      Audio data in bytes or None if synthesis failed.
  """
  if not text:
    logger.error('No text provided for speech synthesis')
    return None

  if not OPENAI_API_KEY:
    logger.error('OpenAI API key not provided')
    return None

  logger.info(f'Converting text to speech using voice: {voice}, model: {model}')
  client = AsyncOpenAI(api_key=OPENAI_API_KEY)

  for attempt in range(max_retries):
    try:
      # Call OpenAI TTS API asynchronously
      response = await client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        speed=speed,
        response_format=output_format,
      )

      # Get audio content as bytes
      audio_data = response.content

      logger.info('Successfully converted text to speech')
      return audio_data

    except Exception as e:
      if attempt == max_retries - 1:
        logger.error(f'Failed after {max_retries} attempts: {e}')
        return None

      wait_time = 2**attempt
      logger.warning(f'Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}')
      await asyncio.sleep(wait_time)


async def synthesize_long_text_async(
  text: str,
  voice: str = DEFAULT_VOICE,
  model: str = TTS_MODEL,
  output_format: str = DEFAULT_OUTPUT_FORMAT,
  speed: float = DEFAULT_SPEED,
  min_chunk_size: int = None,
  concurrency_limit: int = CONCURRENCY_LIMIT,  # Limit concurrent API calls to avoid rate limits
) -> Optional[bytes]:
  """Convert long text to speech asynchronously by breaking it into chunks.

  Args:
      text: The text to convert to speech.
      voice: The voice to use for synthesis.
      model: TTS model to use.
      output_format: The audio format for the output.
      speed: Speech speed multiplier.
      min_chunk_size: Minimum efficient chunk size for TTS API calls.
      concurrency_limit: Maximum number of concurrent API calls.

  Returns:
      Combined audio data in bytes or None if synthesis failed.
  """
  if not text:
    logger.error('No text provided for speech synthesis')
    return None

  logger.info(f'Original text length: {len(text)} characters')

  # Split text into paragraphs
  paragraphs = text.split('\n\n')
  logger.info(f'Number of paragraphs after splitting: {len(paragraphs)}')

  # Log the first few paragraphs to understand structure
  for i, para in enumerate(paragraphs[:5]):
    logger.info(f'Paragraph {i + 1} preview: {para[:100]}... ({len(para)} chars)')

  # Process each paragraph, splitting if needed
  large_paragraphs_count = 0
  raw_chunks = []

  for i, paragraph in enumerate(paragraphs):
    # Skip empty paragraphs
    if not paragraph.strip():
      continue

    # Log if paragraph is large
    if len(paragraph) > MAX_TTS_CHUNK_SIZE:
      large_paragraphs_count += 1
      logger.info(
        f'Large paragraph found (#{i + 1}): {len(paragraph)} chars, exceeds {MAX_TTS_CHUNK_SIZE} limit'  # noqa: E501
      )

    # Split paragraph if it's too large
    paragraph_chunks = split_paragraph_if_needed(paragraph, MAX_TTS_CHUNK_SIZE)

    if len(paragraph_chunks) > 1:
      logger.info(f'Paragraph #{i + 1} was split into {len(paragraph_chunks)} chunks')

    for chunk in paragraph_chunks:
      raw_chunks.append(chunk)

  # Use provided min_chunk_size or fall back to config value
  min_efficient_chunk_size = (
    min_chunk_size if min_chunk_size is not None else MIN_EFFICIENT_CHUNK_SIZE
  )

  # ADVANCED OPTIMIZATION: Intelligently join related paragraphs
  optimized_chunks = []
  current_chunk = ''

  for i, chunk in enumerate(raw_chunks):
    # Check if this chunk is a section marker or heading
    is_section = is_section_marker(chunk)

    # Start a new chunk if:
    # 1. This is a section marker and we've already accumulated enough text, or
    # 2. Adding this chunk would exceed the maximum size
    if (is_section and len(current_chunk) >= min_efficient_chunk_size) or (
      len(current_chunk) + len(chunk) + 2 > MAX_TTS_CHUNK_SIZE
    ):
      if current_chunk:
        optimized_chunks.append(current_chunk)
        current_chunk = chunk
    else:
      # Add to current chunk with a separator if needed
      if current_chunk and chunk:
        current_chunk += '\n\n' + chunk
      else:
        current_chunk = chunk

  # Add the last chunk if not empty
  if current_chunk:
    optimized_chunks.append(current_chunk)

  original_chunk_count = len(raw_chunks)
  optimized_chunk_count = len(optimized_chunks)

  logger.info(
    f'Optimized chunks: {original_chunk_count} paragraphs merged into {optimized_chunk_count} chunks'  # noqa: E501
  )
  logger.info(
    f'Average optimized chunk size: {sum(len(c) for c in optimized_chunks) / len(optimized_chunks) if optimized_chunks else 0:.1f} characters'  # noqa: E501
  )

  # Process chunks asynchronously with a semaphore to limit concurrency
  semaphore = asyncio.Semaphore(concurrency_limit)

  async def process_chunk(i, chunk):
    async with semaphore:
      logger.info(
        f'Processing chunk {i + 1}/{len(optimized_chunks)} ({len(chunk)} chars)'
      )
      audio_data = await synthesize_speech_async(
        chunk, voice=voice, model=model, output_format=output_format, speed=speed
      )
      if not audio_data:
        logger.error(f'Failed to synthesize chunk {i + 1}')
      return i, audio_data

  # Create tasks for all chunks
  tasks = [process_chunk(i, chunk) for i, chunk in enumerate(optimized_chunks)]

  # Run tasks concurrently and wait for them to complete
  results = await asyncio.gather(*tasks)

  # Filter out None results and sort by original index
  audio_chunks = [data for _, data in sorted(results) if data is not None]

  if not audio_chunks:
    logger.error('No audio chunks were successfully synthesized')
    return None

  # Combine all audio chunks
  logger.info(f'Combining {len(audio_chunks)} audio chunks')
  return b''.join(audio_chunks)


def synthesize_long_text(
  text: str,
  voice: str = DEFAULT_VOICE,
  model: str = TTS_MODEL,
  output_format: str = DEFAULT_OUTPUT_FORMAT,
  speed: float = DEFAULT_SPEED,
  min_chunk_size: int = None,
) -> Optional[bytes]:
  """Convert long text to speech by breaking it into chunks.

  This is a synchronous wrapper around the asynchronous implementation.

  Args:
      text: The text to convert to speech.
      voice: The voice to use for synthesis.
      model: TTS model to use.
      output_format: The audio format for the output.
      speed: Speech speed multiplier.
      min_chunk_size: Minimum efficient chunk size for TTS API calls.

  Returns:
      Combined audio data in bytes or None if synthesis failed.
  """
  return asyncio.run(
    synthesize_long_text_async(text, voice, model, output_format, speed, min_chunk_size)
  )
