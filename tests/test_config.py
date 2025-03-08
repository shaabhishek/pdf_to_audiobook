"""Tests for the config module."""

import os
from unittest.mock import patch

import pytest

from pdf_to_audiobook import config


@pytest.fixture
def mock_env_vars():
  """Set up mock environment variables for testing."""
  env_vars = {
    'OPENAI_API_KEY': 'test_openai_key',
    'TTS_MODEL': 'test-model-1',
    'DEFAULT_VOICE': 'test_voice',
    'DEFAULT_SPEED': '1.2',
    'LOG_LEVEL': 'DEBUG',
  }
  with patch.dict(os.environ, env_vars):
    yield env_vars


def test_config_loads_environment_variables(mock_env_vars):
  """Test that config loads environment variables correctly."""
  # Reload the config module to apply the mocked environment variables
  import importlib

  importlib.reload(config)

  # Check if config variables match the mocked environment variables
  assert config.OPENAI_API_KEY == mock_env_vars['OPENAI_API_KEY']
  assert config.TTS_MODEL == mock_env_vars['TTS_MODEL']
  assert config.DEFAULT_VOICE == mock_env_vars['DEFAULT_VOICE']
  assert config.DEFAULT_SPEED == float(mock_env_vars['DEFAULT_SPEED'])
  assert config.LOG_LEVEL == mock_env_vars['LOG_LEVEL']


def test_config_defaults():
  """Test default values when environment variables are not set."""
  # Use a controlled environment with only the variables we're not testing
  # This approach allows us to test defaults while avoiding issues with
  # actual API keys that might be in the environment
  with patch.dict(
    os.environ,
    {
      'OPENAI_API_KEY': 'some_key',  # Not testing this
    },
    clear=True,
  ):
    import importlib

    importlib.reload(config)

    # Check default values for other config variables
    assert config.TTS_MODEL == 'tts-1'
    assert config.DEFAULT_VOICE == 'alloy'
    assert config.DEFAULT_SPEED == 1.25
    assert config.LOG_LEVEL == 'INFO'
