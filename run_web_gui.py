#!/usr/bin/env python
"""Run the PDF to Audiobook Web GUI.

This script provides a simple way to start the web-based GUI from the command line.
"""

import os

# ensure Gradio uses Heroku’s port and host
os.environ.setdefault('GRADIO_SERVER_NAME', '0.0.0.0')
os.environ.setdefault('GRADIO_SERVER_PORT', os.getenv('PORT', '7860'))

from pdf_to_audiobook.web_gui import main

if __name__ == '__main__':
  main()
