"""Tests for the pdf_reader module."""

import base64
import os
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from pdf_to_audiobook import pdf_reader


@pytest.fixture
def sample_pdf_content():
  """Sample PDF content for testing."""
  return b'%PDF-1.7\n...sample PDF content...'


@pytest.fixture
def sample_pdf_base64(sample_pdf_content):
  """Sample PDF content encoded in base64."""
  return base64.b64encode(sample_pdf_content).decode('utf-8')


@pytest.fixture
def mock_openai_response():
  """Mock response from OpenAI API."""
  # Create a simplified mock response structure based on the actual OpenAI response
  mock_message = MagicMock()
  mock_message.content = 'Extracted text from PDF'

  mock_choice = MagicMock()
  mock_choice.message = mock_message

  mock_response = MagicMock()
  mock_response.choices = [mock_choice]
  return mock_response


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


def test_encode_pdf(sample_pdf_content, sample_pdf_base64):
  """Test encoding a PDF file to base64."""
  mock_file = mock_open(read_data=sample_pdf_content)

  with patch('builtins.open', mock_file):
    result = pdf_reader.encode_pdf('test.pdf')

  assert result == sample_pdf_base64
  mock_file.assert_called_once_with('test.pdf', 'rb')


def test_encode_pdf_file_not_found():
  """Test encoding a PDF file that doesn't exist."""
  with patch('builtins.open', side_effect=FileNotFoundError):
    result = pdf_reader.encode_pdf('nonexistent.pdf')

  assert result is None


def test_read_pdf_success(sample_pdf_base64, mock_openai_response):
  """Test successfully reading a PDF file."""
  with (
    patch('os.path.exists', return_value=True),
    patch(
      'pdf_to_audiobook.pdf_reader.extract_text_from_pdf',
      return_value='Raw text from PDF',
    ),
    patch(
      'pdf_to_audiobook.pdf_reader.process_with_gemini',
      return_value='Extracted text from PDF',
    ),
  ):
    result = pdf_reader.read_pdf('test.pdf')

  assert result == 'Extracted text from PDF'


def test_read_pdf_file_not_found():
  """Test reading a PDF file that doesn't exist."""
  with patch('os.path.exists', return_value=False):
    result = pdf_reader.read_pdf('nonexistent.pdf')

  assert result is None


def test_read_pdf_not_pdf():
  """Test reading a file that is not a PDF."""
  with patch('os.path.exists', return_value=True):
    result = pdf_reader.read_pdf('test.txt')

  assert result is None


def test_read_pdf_encoding_error():
  """Test PDF reading when extraction fails."""
  with (
    patch('os.path.exists', return_value=True),
    patch('pdf_to_audiobook.pdf_reader.extract_text_from_pdf', return_value=None),
  ):
    result = pdf_reader.read_pdf('test.pdf')

  assert result is None


def test_read_pdf_api_error(sample_pdf_base64):
  """Test PDF reading when Gemini API call fails."""
  with (
    patch('os.path.exists', return_value=True),
    patch(
      'pdf_to_audiobook.pdf_reader.extract_text_from_pdf',
      return_value='Raw text from PDF',
    ),
    patch('pdf_to_audiobook.pdf_reader.process_with_gemini', return_value=None),
  ):
    result = pdf_reader.read_pdf('test.pdf')

  assert result is None
