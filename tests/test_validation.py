"""Tests for the validation module."""

import argparse
from unittest.mock import patch

import pytest

from pdf_to_audiobook.validation import valid_pdf_file
from pdf_to_audiobook.validation import valid_speed


def test_valid_pdf_file_success():
  """Test valid_pdf_file with a valid PDF file."""
  with (
    patch('os.path.exists', return_value=True),
    patch('os.path.isfile', return_value=True),
    patch('os.access', return_value=True),
  ):
    pdf_path = 'valid.pdf'
    result = valid_pdf_file(pdf_path)
    assert result == pdf_path


def test_valid_pdf_file_not_exist():
  """Test valid_pdf_file with a non-existent file."""
  with patch('os.path.exists', return_value=False):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
      valid_pdf_file('nonexistent.pdf')
    assert 'File does not exist' in str(excinfo.value)


def test_valid_pdf_file_not_a_file():
  """Test valid_pdf_file with a path that is not a file."""
  with (
    patch('os.path.exists', return_value=True),
    patch('os.path.isfile', return_value=False),
  ):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
      valid_pdf_file('directory.pdf')
    assert 'Not a file' in str(excinfo.value)


def test_valid_pdf_file_not_a_pdf():
  """Test valid_pdf_file with a non-PDF file."""
  with (
    patch('os.path.exists', return_value=True),
    patch('os.path.isfile', return_value=True),
  ):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
      valid_pdf_file('notapdf.txt')
    assert 'Not a PDF file' in str(excinfo.value)


def test_valid_pdf_file_not_readable():
  """Test valid_pdf_file with a file that is not readable."""
  with (
    patch('os.path.exists', return_value=True),
    patch('os.path.isfile', return_value=True),
    patch('os.access', return_value=False),
  ):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
      valid_pdf_file('unreadable.pdf')
    assert 'File is not readable' in str(excinfo.value)


def test_valid_speed_success():
  """Test valid_speed with valid values."""
  test_cases = ['0.25', '1.0', '2.5', '4.0']
  for speed in test_cases:
    result = valid_speed(speed)
    assert result == float(speed)


def test_valid_speed_invalid_format():
  """Test valid_speed with invalid format."""
  with pytest.raises(argparse.ArgumentTypeError) as excinfo:
    valid_speed('not-a-number')
  assert 'Invalid speed value' in str(excinfo.value)


def test_valid_speed_out_of_range_low():
  """Test valid_speed with value below the minimum."""
  with pytest.raises(argparse.ArgumentTypeError) as excinfo:
    valid_speed('0.1')
  assert 'Speed must be between 0.25 and 4.0' in str(excinfo.value)


def test_valid_speed_out_of_range_high():
  """Test valid_speed with value above the maximum."""
  with pytest.raises(argparse.ArgumentTypeError) as excinfo:
    valid_speed('5.0')
  assert 'Speed must be between 0.25 and 4.0' in str(excinfo.value)
