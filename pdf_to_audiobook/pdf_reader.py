"""PDF Reader module for extracting text from research papers.

This module uses Google's Gemini API or OpenAI API to extract text from PDF files,
specifically optimized for research papers.
"""

import base64
import logging
import os
import time
from typing import Literal
from typing import Optional

import google.generativeai as genai
import openai
from pypdf import PdfReader

from pdf_to_audiobook.config import DEFAULT_PDF_AI_MODEL
from pdf_to_audiobook.config import GEMINI_API_KEY
from pdf_to_audiobook.config import GEMINI_EXTRACTION_PROMPT
from pdf_to_audiobook.config import GEMINI_MODEL
from pdf_to_audiobook.config import LOG_LEVEL
from pdf_to_audiobook.config import MAX_API_RETRIES
from pdf_to_audiobook.config import MAX_PDF_PROCESSING_SIZE
from pdf_to_audiobook.config import OPENAI_API_KEY
from pdf_to_audiobook.config import OPENAI_EXTRACTION_PROMPT
from pdf_to_audiobook.config import OPENAI_PDF_MODEL

# Configure logging
logging.basicConfig(
  level=getattr(logging, LOG_LEVEL),
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize Gemini API configuration
genai.configure(api_key=GEMINI_API_KEY)

# Create a reusable Gemini model instance
_gemini_model = None


def get_gemini_model():
  """Get or create a singleton instance of the Gemini model.

  Returns:
      A GenerativeModel instance.
  """
  global _gemini_model
  if _gemini_model is None:
    _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
  return _gemini_model


def encode_pdf(file_path: str) -> Optional[str]:
  """Encode a PDF file to base64 for API transmission.

  Args:
      file_path: Path to the PDF file.

  Returns:
      Base64 encoded string of the PDF file or None if encoding failed.
  """
  try:
    with open(file_path, 'rb') as f:
      pdf_bytes = f.read()
    return base64.b64encode(pdf_bytes).decode('utf-8')
  except Exception as e:
    logger.error(f'Error encoding PDF file: {e}')
    return None


def extract_text_from_pdf(file_path: str) -> Optional[str]:
  """Extract raw text from a PDF file using PyPDF.

  Args:
      file_path: Path to the PDF file.

  Returns:
      Raw text extracted from the PDF or None if extraction failed.
  """
  try:
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
      page_text = page.extract_text()
      if page_text:
        text += page_text + '\n\n'
    return text
  except Exception as e:
    logger.error(f'Error extracting raw text from PDF: {e}')
    return None


def process_with_openai_with_retry(
  text: str, max_retries: int = MAX_API_RETRIES
) -> Optional[str]:
  """Process text using OpenAI API with retry logic.

  Args:
      text: Raw text extracted from PDF.
      max_retries: Maximum number of retry attempts.

  Returns:
      Formatted text suitable for audiobook or None if processing failed.
  """
  logger.info(f'Processing text with OpenAI API ({len(text)} chars)')

  if not OPENAI_API_KEY:
    logger.error('OpenAI API key not provided')
    return None

  for attempt in range(max_retries):
    try:
      # Replace the placeholder in the prompt with the actual paper content
      prompt = OPENAI_EXTRACTION_PROMPT.replace('{RESEARCH_PAPER_CONTENT}', text)

      logger.info(
        f'Sending prompt to OpenAI API (total prompt length: {len(prompt)} chars)'
      )

      # Create OpenAI client
      client = openai.OpenAI(api_key=OPENAI_API_KEY)

      # Generate response using the OpenAI API
      logger.info('Waiting for OpenAI API response...')

      # Create base parameters without temperature
      params = {
        'model': OPENAI_PDF_MODEL,
        'messages': [
          {
            'role': 'system',
            'content': 'You are an expert at extracting and formatting research papers for audiobooks.',
          },
          {'role': 'user', 'content': prompt},
        ],
      }
      if OPENAI_PDF_MODEL == 'o3-mini':
        params['reasoning_effort'] = 'medium'
      # Only add temperature for models that support it (not o3-mini)
      if 'o3-mini' not in OPENAI_PDF_MODEL:
        params['temperature'] = 0.2  # Lower temperature for more deterministic results

      # Make API call with appropriate parameters
      response = client.chat.completions.create(**params)

      # Get the formatted text from the response
      formatted_text = response.choices[0].message.content
      logger.info(f'Received response from OpenAI API ({len(formatted_text)} chars)')

      # Log the structure of the response
      lines = formatted_text.split('\n')
      section_count = 0
      for line in lines[:20]:  # Look at first 20 lines
        if '[pause]' in line:
          section_count += 1
          logger.info(f'Found section break: "{line}"')
        elif line.strip() and ':' in line and len(line) < 50:
          logger.info(f'Possible heading found: "{line}"')

      if section_count > 0:
        logger.info(
          f'Found approximately {section_count} section breaks in the first 20 lines'
        )

      # Count total [pause] markers in the response
      total_pauses = formatted_text.count('[pause]')
      logger.info(f'Total [pause] markers in the response: {total_pauses}')

      # Return the formatted text
      return formatted_text
    except Exception as e:
      if attempt == max_retries - 1:
        logger.error(f'Failed after {max_retries} attempts: {e}')
        return None

      wait_time = 2**attempt
      logger.warning(f'Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}')
      time.sleep(wait_time)


def process_with_openai(text: str) -> Optional[str]:
  """Process text using OpenAI API to format for audiobook.

  Args:
      text: Raw text extracted from PDF.

  Returns:
      Formatted text suitable for audiobook or None if processing failed.
  """
  return process_with_openai_with_retry(text)


def process_with_gemini_with_retry(
  text: str, max_retries: int = MAX_API_RETRIES
) -> Optional[str]:
  """Process text using Gemini API with retry logic.

  Args:
      text: Raw text extracted from PDF.
      max_retries: Maximum number of retry attempts.

  Returns:
      Formatted text suitable for audiobook or None if processing failed.
  """
  logger.info(f'Processing text with Gemini API ({len(text)} chars)')

  for attempt in range(max_retries):
    try:
      # Use the singleton model instance
      model = get_gemini_model()

      # Replace the placeholder in the prompt with the actual paper content
      prompt = GEMINI_EXTRACTION_PROMPT.replace('{RESEARCH_PAPER_CONTENT}', text)

      logger.info(
        f'Sending prompt to Gemini API (total prompt length: {len(prompt)} chars)'
      )

      # Generate response
      logger.info('Waiting for Gemini API response...')
      response = model.generate_content(prompt)

      # Get and log the formatted text
      formatted_text = response.text
      logger.info(f'Received response from Gemini API ({len(formatted_text)} chars)')

      # Log the structure of the response
      lines = formatted_text.split('\n')
      section_count = 0
      for line in lines[:20]:  # Look at first 20 lines
        if '[pause]' in line:
          section_count += 1
          logger.info(f'Found section break: "{line}"')
        elif line.strip() and ':' in line and len(line) < 50:
          logger.info(f'Possible heading found: "{line}"')

      if section_count > 0:
        logger.info(
          f'Found approximately {section_count} section breaks in the first 20 lines'
        )

      # Count total [pause] markers in the response
      total_pauses = formatted_text.count('[pause]')
      logger.info(f'Total [pause] markers in the response: {total_pauses}')

      # Return the formatted text
      return formatted_text
    except Exception as e:
      if attempt == max_retries - 1:
        logger.error(f'Failed after {max_retries} attempts: {e}')
        return None

      wait_time = 2**attempt
      logger.warning(f'Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}')
      time.sleep(wait_time)


def process_with_gemini(text: str) -> Optional[str]:
  """Process text using Gemini API to format for audiobook.

  Args:
      text: Raw text extracted from PDF.

  Returns:
      Formatted text suitable for audiobook or None if processing failed.
  """
  return process_with_gemini_with_retry(text)


def read_pdf(
  file_path: str, ai_model: Literal['gemini', 'openai'] = DEFAULT_PDF_AI_MODEL
) -> Optional[str]:
  """Extract and format text from a PDF file for audiobook conversion.

  Args:
      file_path: Path to the PDF file.
      ai_model: AI model to use for processing ('gemini' or 'openai').

  Returns:
      Formatted text suitable for audiobook or None if extraction failed.
  """
  if not os.path.exists(file_path):
    logger.error(f'PDF file not found: {file_path}')
    return None

  if not file_path.lower().endswith('.pdf'):
    logger.error(f'File is not a PDF: {file_path}')
    return None

  logger.info(f'Extracting text from PDF: {file_path}')

  # Extract raw text from PDF
  raw_text = extract_text_from_pdf(file_path)
  if not raw_text:
    logger.error(f'Failed to extract raw text from PDF: {file_path}')
    return None

  # Count paragraphs in raw text
  raw_paragraphs = raw_text.split('\n\n')
  logger.info(
    f'Successfully extracted raw text ({len(raw_text)} characters, {len(raw_paragraphs)} paragraphs)'
  )

  # Sample the raw text to understand its structure
  logger.info(f'Raw text sample (first 200 chars): {raw_text[:200]}')

  # Handle large PDFs by chunking if necessary
  if len(raw_text) > MAX_PDF_PROCESSING_SIZE:
    logger.info(f'Document is large ({len(raw_text)} chars), using chunked processing')
    # For large documents, we might want to implement a more sophisticated chunking strategy
    # For now, we'll just take the first MAX_PDF_PROCESSING_SIZE characters (or less if smaller)
    processing_text = raw_text[: min(len(raw_text), MAX_PDF_PROCESSING_SIZE)]
    logger.info(
      f'Processing first {len(processing_text)} characters ({len(processing_text.split("\n\n"))} paragraphs)'
    )
  else:
    processing_text = raw_text

  # Process with selected AI model
  logger.info(f'Using {ai_model.upper()} model for text processing')

  if ai_model.lower() == 'openai':
    formatted_text = process_with_openai(processing_text)
  else:  # Default to gemini
    formatted_text = process_with_gemini(processing_text)

  if not formatted_text:
    logger.error(f'Failed to process text with {ai_model.upper()} API')
    return None

  # Analyze the formatted text structure
  formatted_paragraphs = formatted_text.split('\n\n')
  logger.info(
    f'Successfully processed text with {ai_model.upper()} API ({len(formatted_text)} characters, {len(formatted_paragraphs)} paragraphs)'
  )

  # Count empty lines and potential section breaks
  empty_lines = formatted_text.count('\n\n')
  section_breaks = formatted_text.count('[pause]')

  logger.info(
    f'Formatted text statistics: {empty_lines} paragraph breaks, {section_breaks} section breaks'
  )
  logger.info(f'Formatted text sample (first 200 chars): {formatted_text[:200]}')

  return formatted_text
