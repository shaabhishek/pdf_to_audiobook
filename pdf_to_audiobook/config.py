"""Configuration settings for the PDF to Audiobook converter.

This module contains API keys, endpoints, and other configuration settings.
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# OpenAI model for PDF text extraction
# Note: o3-mini doesn't support temperature parameter and has some limitations
# compared to more powerful models like gpt-3.5-turbo or gpt-4
OPENAI_PDF_MODEL = os.getenv('OPENAI_PDF_MODEL', 'o3-mini')

# Google Gemini API configuration (used for PDF text extraction)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

# Default AI model to use for PDF processing (either 'gemini' or 'openai')
DEFAULT_PDF_AI_MODEL = os.getenv('DEFAULT_PDF_AI_MODEL', 'gemini')

# OpenAI TTS API configuration
# No separate API key needed for OpenAI TTS - uses the same OPENAI_API_KEY
# Default to tts-1 model
TTS_MODEL = os.getenv('TTS_MODEL', 'tts-1')
# OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'alloy')
DEFAULT_OUTPUT_FORMAT = 'mp3'
# Default playback speed multiplier
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', '1.0'))

# Default output filename (used when no output path is specified)
DEFAULT_OUTPUT_FILENAME = 'output_audiobook.mp3'

# TTS API limits and settings
MAX_TTS_CHUNK_SIZE = 4_000  # Maximum chunk size for TTS API (in characters)
# Minimum efficient chunk size for TTS API to reduce API calls
MIN_EFFICIENT_CHUNK_SIZE = 3_000

# PDF processing settings
MAX_PDF_PROCESSING_SIZE = 100_000  # Maximum size of PDF text to process

# API retry settings
MAX_API_RETRIES = 3  # Maximum number of retries for API calls
CONCURRENCY_LIMIT = 5  # Maximum number of concurrent API calls

# OpenAI API prompt for PDF extraction
OPENAI_EXTRACTION_PROMPT = """
You are an expert at extracting and summarizing content from research papers for audiobook conversion. 
Your task is to process the following research paper content and prepare it for audio narration:

<research_paper>
{RESEARCH_PAPER_CONTENT}
</research_paper>

Please follow these steps to extract and format the content:

1. Read through the entire paper carefully.

2. Extract and structure the content as follows:
   a. Title and Authors: Present the paper's title and list of authors but not their affiliations.
   b. Abstract: Include the full abstract.
   c. Main Sections: Extract the key content from the introduction, methodology, results, and conclusion. Omit any references, citations, and footnotes.
   d. Mathematical Formulas: Convert equations and formulas into spoken language when possible. For complex formulas that cannot be easily verbalized, provide a brief description of their purpose or significance.
   e. Figures and Tables: Briefly describe important visual elements, focusing on their key insights and relevance to the overall research.

3. Format the text for natural, engaging audio narration:
   a. Use clear transitions between sections.
   b. Break down long, complex sentences into more digestible parts.
   c. Spell out acronyms and abbreviations on first use.
   d. Assume the reader is a scientist or engineer with a strong background in the subject matter.
   e. Use the active voice, assuming that you are narrating YOUR OWN paper to the reader.

4. Maintain the academic tone while optimizing for listening comprehension:
   a. Use clear, concise language.
   b. Emphasize key findings and their implications.
   c. Provide context for technical terms or concepts when necessary.

5. Add [pause] marker where there should be a pause in the audio, especially between major sections.

Remember to focus on creating a coherent, engaging narrative that captures the essence of the research paper in a format suitable for audio presentation.

Aim for a total length of 15000-20000 characters.
"""  # noqa: E501

# Gemini API prompt for PDF extraction
GEMINI_EXTRACTION_PROMPT = """
You are an expert at extracting and summarizing content from research papers for audiobook conversion. 
Your task is to process the following research paper content and prepare it for audio narration:

<research_paper>
{RESEARCH_PAPER_CONTENT}
</research_paper>

Please follow these steps to extract and format the content:

1. Read through the entire paper carefully.

2. Extract and structure the content as follows:
   a. Title and Authors: Present the paper's title and list of authors but not their affiliations.
   b. Abstract: Include the full abstract.
   c. Main Sections: Extract the key content from the introduction, methodology, results, and conclusion. Omit any references, citations, and footnotes.
   d. Mathematical Formulas: Convert equations and formulas into spoken language when possible. For complex formulas that cannot be easily verbalized, provide a brief description of their purpose or significance.
   e. Figures and Tables: Briefly describe important visual elements, focusing on their key insights and relevance to the overall research.

3. Format the text for natural, engaging audio narration:
   a. Use clear transitions between sections.
   b. Break down long, complex sentences into more digestible parts.
   c. Spell out acronyms and abbreviations on first use.
   d. Assume the reader is a scientist or engineer with a strong background in the subject matter.
   e. Use the active voice, assuming that you are narrating YOUR OWN paper to the reader.

4. Maintain the academic tone while optimizing for listening comprehension:
   a. Use clear, concise language.
   b. Emphasize key findings and their implications.
   c. Provide context for technical terms or concepts when necessary.

5. Add [pause] marker where there should be a pause in the audio, especially between major sections.

Remember to focus on creating a coherent, engaging narrative that captures the essence of the research paper in a format suitable for audio presentation.

Aim for a total length of 15000-20000 characters.
"""  # noqa: E501

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
