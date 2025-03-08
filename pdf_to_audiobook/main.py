"""Main module for PDF to Audiobook converter.

This module provides the command-line interface for converting
PDF research papers to audiobooks.
"""

import argparse
import contextlib
import pathlib
import time
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional

from pdf_to_audiobook.config import DEFAULT_PDF_AI_MODEL
from pdf_to_audiobook.config import DEFAULT_SPEED
from pdf_to_audiobook.config import DEFAULT_VOICE
from pdf_to_audiobook.config import TEXT_OUTPUT_DIR
from pdf_to_audiobook.config import TTS_MODEL
from pdf_to_audiobook.file_utils import ensure_directory_exists
from pdf_to_audiobook.file_utils import save_binary_to_file
from pdf_to_audiobook.file_utils import save_text_to_file
from pdf_to_audiobook.logging_config import configure_logging
from pdf_to_audiobook.pdf_reader import read_pdf
from pdf_to_audiobook.tts_client import synthesize_long_text
from pdf_to_audiobook.utils import get_snake_case_title_from_pdf
from pdf_to_audiobook.utils import to_snake_case
from pdf_to_audiobook.validation import valid_pdf_file
from pdf_to_audiobook.validation import valid_speed

# Configure logging
logger = configure_logging(__name__)


class OutputPaths(NamedTuple):
  """Container for output file paths."""

  text_path: pathlib.Path
  audio_path: pathlib.Path
  title: str


def get_output_paths(
  pdf_path: str,
  ai_model: str,
  output_folder: Optional[str] = None,
  custom_title: Optional[str] = None,
) -> OutputPaths:
  """Generate output paths for both text and audio files.

  Args:
      pdf_path: Path to the PDF file.
      ai_model: AI model used for PDF processing.
      output_folder: Optional custom output folder.
      custom_title: Optional custom title to use for filenames.

  Returns:
      OutputPaths containing text_path, audio_path and title.
  """
  # Convert to Path object for easier manipulation
  pdf_path_obj = pathlib.Path(pdf_path).resolve()

  # Get snake case title
  if custom_title:
    snake_case_title = to_snake_case(custom_title)
    logger.info(f'Using provided custom title: {custom_title}')
  else:
    snake_case_title = get_snake_case_title_from_pdf(pdf_path)

  # Generate filenames with model name
  text_filename = f'{snake_case_title}_{ai_model}.md'
  audio_filename = f'{snake_case_title}_{ai_model}.mp3'

  # Determine output directory
  if output_folder:
    output_dir = pathlib.Path(output_folder)
  else:
    # Use TEXT_OUTPUT_DIR for text and pdf's directory for audio
    text_dir = pathlib.Path(TEXT_OUTPUT_DIR)
    audio_dir = pdf_path_obj.parent

  # Construct paths
  if output_folder:
    ensure_directory_exists(output_folder)
    text_path = output_dir / text_filename
    audio_path = output_dir / audio_filename
  else:
    ensure_directory_exists(TEXT_OUTPUT_DIR)
    text_path = text_dir / text_filename
    audio_path = audio_dir / audio_filename

  return OutputPaths(text_path, audio_path, snake_case_title)


def process_pdf(
  pdf_path: str,
  output_paths: OutputPaths,
  ai_model: str,
) -> Optional[str]:
  """Extract text from PDF and save it.

  Args:
      pdf_path: Path to the PDF file.
      output_paths: Paths for output files.
      ai_model: AI model to use for PDF processing.

  Returns:
      Extracted text if successful, None otherwise.
  """
  logger.info(f'Using {ai_model.upper()} model for PDF text extraction')
  extracted_text = read_pdf(pdf_path, ai_model=ai_model)

  if not extracted_text:
    logger.error('Failed to extract text from PDF')
    return None

  logger.info(
    f'Successfully extracted text from PDF ({len(extracted_text)} characters)'
  )

  # Save the text to a markdown file
  saved_path = save_text_to_file(extracted_text, str(output_paths.text_path))

  if saved_path:
    logger.info(f'Successfully saved text version to {saved_path}')
  else:
    logger.warning('Failed to save text version, but continuing with audio conversion')

  return extracted_text


def convert_text_to_audio(
  text: str,
  output_path: pathlib.Path,
  voice: str = DEFAULT_VOICE,
  tts_model: str = TTS_MODEL,
  speed: float = DEFAULT_SPEED,
  min_chunk_size: Optional[int] = None,
) -> bool:
  """Convert text to speech and save as an audio file.

  Args:
      text: The text to convert to speech.
      output_path: Path to save the output audio file.
      voice: Voice to use for the audiobook.
      tts_model: TTS model to use.
      speed: Speech speed multiplier.
      min_chunk_size: Minimum efficient chunk size for TTS API calls.

  Returns:
      True if conversion was successful, False otherwise.
  """
  # Convert text to speech
  logger.info(
    f'Converting text to speech using OpenAI TTS with voice: {voice}, model: {tts_model}'
  )
  audio_data = synthesize_long_text(
    text,
    voice=voice,
    tts_model=tts_model,
    speed=speed,
    min_chunk_size=min_chunk_size,
  )

  if not audio_data:
    logger.error('Failed to convert text to speech')
    return False

  logger.info(f'Successfully generated audio ({len(audio_data)} bytes)')

  # Save audio data to file
  saved_path = save_binary_to_file(audio_data, str(output_path))
  if not saved_path:
    logger.error('Failed to save audio file')
    return False

  file_size_mb = output_path.stat().st_size / (1024 * 1024)
  logger.info(f'Successfully saved audiobook to {saved_path} ({file_size_mb:.2f} MB)')
  return True


def convert_pdf_to_audiobook(
  pdf_path: str,
  output_folder: Optional[str] = None,
  voice: str = DEFAULT_VOICE,
  tts_model: str = TTS_MODEL,
  speed: float = DEFAULT_SPEED,
  min_chunk_size: Optional[int] = None,
  ai_model: str = DEFAULT_PDF_AI_MODEL,
  custom_title: Optional[str] = None,
) -> bool:
  """Convert a PDF file to an audiobook.

  Args:
      pdf_path: Path to the PDF file.
      output_folder: Directory to save output files.
      voice: Voice to use for the audiobook.
      tts_model: TTS model to use.
      speed: Speech speed multiplier.
      min_chunk_size: Minimum efficient chunk size for TTS API calls.
      ai_model: AI model to use for PDF processing.
      custom_title: Optional custom title for output files.

  Returns:
      True if conversion was successful, False otherwise.
  """
  # Get paths for output files
  output_paths = get_output_paths(pdf_path, ai_model, output_folder, custom_title)

  logger.info(f'Starting conversion of {pdf_path} to audiobook')
  logger.info(f'Output file will be saved to: {output_paths.audio_path.absolute()}')

  # Extract text from PDF and save as markdown
  extracted_text = process_pdf(pdf_path, output_paths, ai_model)
  if not extracted_text:
    return False

  # Convert text to speech and save as audio file
  return convert_text_to_audio(
    extracted_text,
    output_paths.audio_path,
    voice=voice,
    tts_model=tts_model,
    speed=speed,
    min_chunk_size=min_chunk_size,
  )


def parse_arguments() -> Dict[str, Any]:
  """Parse command line arguments.

  Returns:
      Dictionary of parsed arguments.
  """
  parser = argparse.ArgumentParser(description='Convert a PDF to an audiobook')

  # Add arguments
  parser.add_argument('pdf_path', type=valid_pdf_file, help='Path to the PDF file')
  parser.add_argument(
    '--output-folder',
    '-o',
    dest='output',
    help='Output folder to save generated files (audio and text)',
  )
  parser.add_argument(
    '--voice',
    choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
    default=DEFAULT_VOICE,
    help='Voice to use for TTS (default: alloy)',
  )
  parser.add_argument(
    '--tts-model',
    choices=['tts-1', 'tts-1-hd'],
    default=TTS_MODEL,
    help='Model to use for TTS (default: tts-1)',
  )
  parser.add_argument(
    '--speed',
    type=valid_speed,
    default=DEFAULT_SPEED,
    help='Speed of speech (0.25 to 4.0, default: 1.25)',
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
  parser.add_argument(
    '--title',
    type=str,
    help='Custom title to use for the output files (will be converted to snake_case)',
  )

  return vars(parser.parse_args())


def main() -> int:
  """Main function for the PDF to audiobook converter."""
  # Parse command line arguments
  args = parse_arguments()

  # Process PDF to audiobook
  success = convert_pdf_to_audiobook(
    pdf_path=args['pdf_path'],
    output_folder=args['output'],
    voice=args['voice'],
    tts_model=args['tts_model'],
    speed=args['speed'],
    min_chunk_size=args['min_chunk_size'],
    ai_model=args['ai_model'],
    custom_title=args['title'],
  )

  if success:
    # Get output path for display
    output_paths = get_output_paths(
      args['pdf_path'], args['ai_model'], args['output'], args['title']
    )

    file_size_mb = output_paths.audio_path.stat().st_size / (1024 * 1024)
    print(f'✅ Successfully converted {args["pdf_path"]} to audiobook:')
    print(
      f'📁 Output file: {output_paths.audio_path.absolute()} ({file_size_mb:.2f} MB)'
    )
    return 0
  else:
    print('❌ Failed to convert PDF to audiobook. Check logs for details.')
    return 1


@contextlib.contextmanager
def time_it():
  """Time the execution of a block of code."""
  start_time = time.time()
  try:
    yield
  finally:
    end_time = time.time()
    print(f'Time taken: {end_time - start_time:.2f} seconds')


if __name__ == '__main__':
  with time_it():
    exit(main())
