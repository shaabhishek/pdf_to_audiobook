"""Web-based GUI for PDF to Audiobook converter.

This module provides a web-based graphical user interface for the PDF to
Audiobook converter using Gradio, which makes it accessible through a web
browser.
"""

import os
import pathlib
import shutil
import tempfile
from typing import Optional
from typing import Tuple

import gradio as gr

from pdf_to_audiobook.config import AI_MODEL_CHOICES
from pdf_to_audiobook.config import DEFAULT_PDF_AI_MODEL
from pdf_to_audiobook.config import DEFAULT_SPEED
from pdf_to_audiobook.config import DEFAULT_VOICE
from pdf_to_audiobook.config import TTS_MODEL
from pdf_to_audiobook.config import TTS_MODEL_CHOICES
from pdf_to_audiobook.config import TTS_VOICE_CHOICES
from pdf_to_audiobook.logging_config import configure_logging
from pdf_to_audiobook.main import convert_pdf_to_audiobook
from pdf_to_audiobook.main import get_output_paths

# Configure logging
logger = configure_logging(__name__)

# Keep track of common output folders to add to allowed_paths
COMMON_OUTPUT_FOLDERS = set()


def get_default_output_folder(pdf_file: str) -> str:
  """Get the default output folder based on PDF location.

  Args:
      pdf_file: Path to the PDF file.

  Returns:
      The parent directory of the PDF file.
  """
  if not pdf_file:
    return ''

  pdf_path = pathlib.Path(pdf_file)
  folder = str(pdf_path.parent)
  # Add to common folders for allowed_paths
  if folder:
    COMMON_OUTPUT_FOLDERS.add(folder)
  return folder


def process_pdf_file(
  pdf_file: str,
  output_folder: str,
  custom_title: str,
  voice: str,
  tts_model: str,
  speed: float,
  min_chunk_size: int,
  ai_model: str,
) -> Tuple[str, Optional[str]]:
  """Process a PDF file and convert it to an audiobook.

  Args:
      pdf_file: Path to the PDF file.
      output_folder: Directory to save output files.
      custom_title: Optional custom title for output files.
      voice: Voice to use for the audiobook.
      tts_model: TTS model to use.
      speed: Speech speed multiplier.
      min_chunk_size: Minimum efficient chunk size for TTS API calls.
      ai_model: AI model to use for PDF processing.

  Returns:
      A tuple containing (status_message, audio_file_path).
  """
  try:
    # Convert empty strings to None
    custom_title_val = custom_title if custom_title.strip() else None
    output_folder_val = output_folder if output_folder.strip() else None
    min_chunk_size_val = int(min_chunk_size) if min_chunk_size > 0 else None

    # Add output folder to common folders if specified
    if output_folder_val:
      COMMON_OUTPUT_FOLDERS.add(output_folder_val)

    # Get output paths using the same logic as main.py
    output_paths = get_output_paths(
      pdf_path=pdf_file,
      ai_model=ai_model,
      output_folder=output_folder_val,
      custom_title=custom_title_val,
    )

    logger.info(f'Starting conversion with TTS model: {tts_model}, voice: {voice}')
    logger.info(f'Output text will be saved to: {output_paths.text_path}')
    logger.info(f'Output audio will be saved to: {output_paths.audio_path}')

    # Process the PDF file
    success = convert_pdf_to_audiobook(
      pdf_path=pdf_file,
      output_folder=output_folder_val,
      voice=voice,
      tts_model=tts_model,
      speed=float(speed),
      min_chunk_size=min_chunk_size_val,
      ai_model=ai_model,
      custom_title=custom_title_val,
    )

    if success:
      # Copy the audio file to a temporary location that Gradio can access
      temp_dir = tempfile.mkdtemp()
      temp_audio_path = os.path.join(
        temp_dir, os.path.basename(output_paths.audio_path)
      )

      # Make sure the output file exists before trying to copy it
      if os.path.exists(output_paths.audio_path):
        logger.info(
          f'Copying audio file from {output_paths.audio_path} to {temp_audio_path}'
        )
        shutil.copy2(output_paths.audio_path, temp_audio_path)
        return (
          f'✅ Success! Audio saved to {output_paths.audio_path}',
          temp_audio_path,
        )
      else:
        logger.error(f'Output audio file not found at {output_paths.audio_path}')
        return (
          f'⚠️ File created but not found at expected location: {output_paths.audio_path}',
          None,
        )
    else:
      return '❌ Conversion failed. Check the logs for details.', None

  except Exception as e:
    logger.exception(f'Error processing PDF: {e}')
    return f'❌ Error: {str(e)}', None


def update_output_folder(pdf_file: Optional[dict]) -> str:
  """Update the output folder based on the uploaded PDF file.

  Args:
      pdf_file: The uploaded PDF file object from Gradio.

  Returns:
      The default output folder path based on the PDF location.
  """
  if not pdf_file:
    return ''

  # Gradio file objects have a 'name' field with the file path
  pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else ''
  if pdf_path:
    return get_default_output_folder(pdf_path)
  return ''


def create_web_interface():
  """Create the web interface using Gradio."""
  with gr.Blocks(title='PDF to Audiobook Converter') as interface:
    gr.Markdown('# PDF to Audiobook Converter')
    gr.Markdown(
      'Upload a PDF and convert it to an audiobook using text-to-speech technology.'
    )
    logger.info('Creating web interface with the following settings:')
    logger.info('TTS Model: %s', TTS_MODEL)
    logger.info('Default Voice: %s', DEFAULT_VOICE)
    logger.info('Default Speed: %s', DEFAULT_SPEED)
    logger.info('AI Model: %s', DEFAULT_PDF_AI_MODEL)

    with gr.Row():
      with gr.Column(scale=2):
        # Input options
        pdf_file = gr.File(label='PDF File', file_types=['.pdf'])
        output_folder = gr.Textbox(
          label='Output Folder (Optional)',
          placeholder='Leave empty to use default location',
        )
        custom_title = gr.Textbox(
          label='Custom Title (Optional)', placeholder='Leave empty to use PDF filename'
        )

        with gr.Accordion('TTS Settings', open=True):
          voice = gr.Dropdown(
            choices=TTS_VOICE_CHOICES,
            value=DEFAULT_VOICE,
            label='Voice',
          )
          # The default value is set to be the first option in the list
          tts_model = gr.Dropdown(
            choices=TTS_MODEL_CHOICES,
            label='TTS Model',
          )
          speed = gr.Slider(
            minimum=0.25,
            maximum=4.0,
            value=DEFAULT_SPEED,
            step=0.05,
            label='Speed',
          )

        with gr.Accordion('Advanced Settings', open=True):
          min_chunk_size = gr.Number(
            value=0,
            label='Minimum Chunk Size for TTS API (0 = use default)',
            precision=0,
          )
          ai_model = gr.Dropdown(
            choices=AI_MODEL_CHOICES,
            value='openai',  # Change default to OpenAI
            label='AI Model for PDF Processing',
          )

        # Convert button
        convert_btn = gr.Button('Convert PDF to Audiobook', variant='primary')

        # Set up event handlers
        pdf_file.change(
          fn=update_output_folder,
          inputs=[pdf_file],
          outputs=[output_folder],
        )

      with gr.Column(scale=1):
        # Output display
        status = gr.Textbox(label='Status', interactive=False)
        audio_output = gr.Audio(
          label='Generated Audiobook', interactive=False, type='filepath'
        )

    # Set up the conversion function
    convert_btn.click(
      fn=process_pdf_file,
      inputs=[
        pdf_file,
        output_folder,
        custom_title,
        voice,
        tts_model,
        speed,
        min_chunk_size,
        ai_model,
      ],
      outputs=[status, audio_output],
    )

  return interface


def main():
  """Main entry point for the web GUI application."""
  interface = create_web_interface()

  # Add common output folders to allowed_paths to improve reliability
  allowed_paths = list(COMMON_OUTPUT_FOLDERS)

  # Add common system locations that might be used
  system_paths = [
    str(pathlib.Path.home()),
    str(pathlib.Path.home() / 'Documents'),
    str(pathlib.Path.home() / 'Downloads'),
    str(pathlib.Path.home() / 'Desktop'),
    str(pathlib.Path.home() / 'Library' / 'CloudStorage'),
  ]
  allowed_paths.extend(system_paths)

  # Launch the interface with allowed paths
  logger.info(f'Launching web interface with allowed paths: {allowed_paths}')
  port = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
  interface.launch(
    server_name=os.getenv('GRADIO_SERVER_NAME', '0.0.0.0'),
    server_port=port,
    share=False,
    allowed_paths=allowed_paths,
  )
  return 0


if __name__ == '__main__':
  main()
