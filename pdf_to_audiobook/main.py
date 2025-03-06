"""Main module for PDF to Audiobook converter.

This module provides the command-line interface for converting
PDF research papers to audiobooks.
"""

import argparse
import atexit
import logging
import os
import pathlib
import time

from pdf_to_audiobook.config import DEFAULT_PDF_AI_MODEL
from pdf_to_audiobook.config import DEFAULT_SPEED
from pdf_to_audiobook.config import DEFAULT_VOICE
from pdf_to_audiobook.config import LOG_LEVEL
from pdf_to_audiobook.config import TTS_MODEL
from pdf_to_audiobook.pdf_reader import read_pdf
from pdf_to_audiobook.tts_client import synthesize_long_text
from pdf_to_audiobook.validation import valid_pdf_file
from pdf_to_audiobook.validation import valid_speed

# Configure logging
logging.basicConfig(
  level=getattr(logging, LOG_LEVEL),
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Register cleanup function to run at exit
def cleanup():
  """Handle graceful shutdown of resources."""
  # Allow time for any pending operations to complete
  time.sleep(1)


# Register the cleanup function to run at program exit
atexit.register(cleanup)


def get_default_output_path(pdf_path: str) -> str:
  """Generate the default output path based on the input PDF path.

  Args:
      pdf_path: Path to the PDF file.

  Returns:
      The default output path with .mp3 extension in the same directory.
  """
  # Get the absolute path to handle relative paths properly
  abs_path = os.path.abspath(pdf_path)

  # Use pathlib to easily manipulate the path
  pdf_file = pathlib.Path(abs_path)

  # Create a new path with the same name but .mp3 extension
  output_path = pdf_file.with_suffix('.mp3')

  return str(output_path)


def convert_pdf_to_audiobook(
  pdf_path: str,
  output_path: str = None,
  voice: str = DEFAULT_VOICE,
  model: str = TTS_MODEL,
  speed: float = DEFAULT_SPEED,
  min_chunk_size: int = None,
  ai_model: str = DEFAULT_PDF_AI_MODEL,
) -> bool:
  """Convert a PDF file to an audiobook.

  Args:
      pdf_path: Path to the PDF file.
      output_path: Path to save the output MP3 file. If None, uses the same
          path as PDF but with .mp3 extension.
      voice: Voice to use for the audiobook (OpenAI voice names).
      model: TTS model to use.
      speed: Speech speed multiplier.
      min_chunk_size: Minimum efficient chunk size for TTS API calls.
      ai_model: AI model to use for PDF processing ('gemini' or 'openai').

  Returns:
      True if conversion was successful, False otherwise.
  """
  # If no output path is provided, use the same path as the PDF but with .mp3 extension
  if output_path is None:
    output_path = get_default_output_path(pdf_path)

  # Extract text from PDF
  logger.info(f'Starting conversion of {pdf_path} to audiobook')
  logger.info(f'Output file will be saved to: {os.path.abspath(output_path)}')
  logger.info(f'Using {ai_model.upper()} model for PDF text extraction')

  extracted_text = read_pdf(pdf_path, ai_model=ai_model)

  if not extracted_text:
    logger.error('Failed to extract text from PDF')
    return False

  logger.info(
    f'Successfully extracted text from PDF ({len(extracted_text)} characters)'
  )

  # Convert text to speech
  logger.info(
    f'Converting text to speech using OpenAI TTS with voice: {voice}, model: {model}'
  )
  audio_data = synthesize_long_text(
    extracted_text, voice=voice, model=model, speed=speed, min_chunk_size=min_chunk_size
  )

  if not audio_data:
    logger.error('Failed to convert text to speech')
    return False

  logger.info(f'Successfully generated audio ({len(audio_data)} bytes)')

  # Save audio data to file
  try:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'wb') as f:
      f.write(audio_data)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
      f'Successfully saved audiobook to {os.path.abspath(output_path)} ({file_size_mb:.2f} MB)'  # noqa: E501
    )
    return True
  except Exception as e:
    logger.error(f'Error saving audiobook: {e}')
    return False


def main() -> int:
  """Main function for the PDF to audiobook converter."""
  # Set up argument parser
  parser = argparse.ArgumentParser(description='Convert a PDF to an audiobook')

  # Add arguments
  parser.add_argument('pdf_path', type=valid_pdf_file, help='Path to the PDF file')
  parser.add_argument('--output', '-o', help='Output audio file path')
  parser.add_argument(
    '--voice',
    choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
    default='alloy',
    help='Voice to use for TTS (default: alloy)',
  )
  parser.add_argument(
    '--model',
    choices=['tts-1', 'tts-1-hd'],
    default='tts-1',
    help='Model to use for TTS (default: tts-1)',
  )
  parser.add_argument(
    '--speed',
    type=valid_speed,
    default=1.0,
    help='Speed of speech (0.25 to 4.0, default: 1.0)',
  )
  parser.add_argument(
    '--min-chunk-size', type=int, help='Minimum chunk size for TTS processing'
  )
  parser.add_argument(
    '--ai-model',
    choices=['gemini', 'openai'],
    default=DEFAULT_PDF_AI_MODEL,
    help='AI model to use for PDF processing (default: %(default)s)',
  )

  # Parse arguments
  args = parser.parse_args()

  # Get args
  pdf_path = args.pdf_path
  output_path = args.output or get_default_output_path(pdf_path)
  voice = args.voice
  model = args.model
  speed = args.speed
  min_chunk_size = args.min_chunk_size
  ai_model = args.ai_model

  # Process PDF to audiobook
  success = convert_pdf_to_audiobook(
    pdf_path=pdf_path,
    output_path=output_path,
    voice=voice,
    model=model,
    speed=speed,
    min_chunk_size=min_chunk_size,
    ai_model=ai_model,
  )

  if success:
    # If output was None, get the default path that was used
    output_path = args.output or get_default_output_path(pdf_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'✅ Successfully converted {pdf_path} to audiobook:')
    print(f'📁 Output file: {os.path.abspath(output_path)} ({file_size_mb:.2f} MB)')
    return 0
  else:
    print('❌ Failed to convert PDF to audiobook. Check logs for details.')
    return 1


if __name__ == '__main__':
  exit(main())
