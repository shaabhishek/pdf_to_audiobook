"""Tests for the utils module."""

import unittest

from pdf_to_audiobook.utils import to_snake_case


class TestUtils(unittest.TestCase):
  """Test cases for the utils module."""

  def test_to_snake_case(self):
    """Test the to_snake_case function."""
    test_cases = [
      ('', ''),
      ('Hello World', 'hello_world'),
      ('Hello, World!', 'hello_world'),
      ('Hello  World', 'hello_world'),
      ('Hello-World', 'hello_world'),
      ('Hello_World', 'hello_world'),
      ('HELLO WORLD', 'hello_world'),
      ('hello world', 'hello_world'),
      ('Hello World 123', 'hello_world_123'),
      ('   Hello   World   ', 'hello_world'),
      ('Hello___World', 'hello_world'),
      ('A Novel Approach to Machine Learning', 'a_novel_approach_to_machine_learning'),
      ('Deep Learning: A Comprehensive Survey', 'deep_learning_a_comprehensive_survey'),
    ]

    for input_text, expected_output in test_cases:
      with self.subTest(input_text=input_text):
        self.assertEqual(to_snake_case(input_text), expected_output)


if __name__ == '__main__':
  unittest.main()
