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
AI_MODEL_CHOICES = ['openai', 'gemini']
DEFAULT_PDF_AI_MODEL = os.getenv('DEFAULT_PDF_AI_MODEL')

# OpenAI TTS API configuration
# No separate API key needed for OpenAI TTS - uses the same OPENAI_API_KEY
# Possible values: tts-1, tts-1-hd
TTS_MODEL_CHOICES = ['tts-1', 'tts-1-hd']
TTS_MODEL = os.getenv('TTS_MODEL')
# OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
TTS_VOICE_CHOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE')
DEFAULT_OUTPUT_FORMAT = 'mp3'
# Default playback speed multiplier
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED'))


# Default output filename (used when no output path is specified)
DEFAULT_OUTPUT_FILENAME = 'output_audiobook.mp3'

# Directory for storing text versions of audiobooks
TEXT_OUTPUT_DIR = os.getenv('TEXT_OUTPUT_DIR')

# TTS API limits and settings
MAX_TTS_CHUNK_SIZE = 4_000  # Maximum chunk size for TTS API (in characters)
# Minimum efficient chunk size for TTS API to reduce API calls
MIN_EFFICIENT_CHUNK_SIZE = 3_000

# PDF processing settings
MAX_PDF_PROCESSING_SIZE = 300_000  # Maximum size of PDF text to process

# API retry settings
MAX_API_RETRIES = 3  # Maximum number of retries for API calls
CONCURRENCY_LIMIT = 5  # Maximum number of concurrent API calls

# OpenAI API prompt for PDF extraction
# OPENAI_EXTRACTION_PROMPT = """
# You are an expert at extracting and summarizing content from research papers for audiobook conversion.
# Your task is to process the following research paper content and prepare it for audio narration:

# <research_paper>
# {RESEARCH_PAPER_CONTENT}
# </research_paper>

# Please follow these steps to extract and format the content:

# 1. Read through the entire paper carefully.

# 2. Extract and structure the content as follows:
#    a. Title and Authors: Present the paper's title and list of authors but not their affiliations.
#    b. Abstract: Ignore the abstract.
#    c. Main Sections: Extract the key content from the introduction, related work, methodology, results, and conclusion (if there are any other sections, add them to the result too). Omit any references, citations, and footnotes.
#    d. Mathematical Formulas: Convert equations and formulas into spoken language when possible. For complex formulas that cannot be easily verbalized, provide a brief description of their purpose or significance.
#    e. Figures and Tables: Briefly describe important visual elements, focusing on their key insights and relevance to the overall research.

# 3. Format the text for natural, engaging audio narration:
#    a. Use clear transitions between sections.
#    b. Break down long, complex sentences into more digestible parts.
#    c. Spell out acronyms and abbreviations on first use.
#    d. Assume the reader is a scientist or engineer with a strong background in the subject matter.
#    e. Use the active voice, assuming that you are narrating YOUR OWN paper to the reader.

# 4. Maintain the academic tone while optimizing for listening comprehension:
#    a. Use clear, concise language.
#    b. Emphasize key findings and their implications.
#    c. Provide context for technical terms or concepts when necessary.

# 5. Add [pause] marker where there should be a pause in the audio, especially between major sections.

# Remember to focus on creating a coherent, engaging narrative that captures the essence of the research paper in a format suitable for audio presentation.

# Aim for a total length of 15000-20000 characters.
# """  # noqa: E501


OPENAI_EXTRACTION_PROMPT = """
Imagine you are the original author of this research paper. Your task is to present an in-depth, audiobook-style explanation of your work that leaves your audience—comprising machine learning Ph.Ds and other experts—a complete understanding of every facet of your research. 

Structure your explanation in the following layers:
	1.	Layer 1 – High-Level Overview:
Provide a concise, broad summary of the research. Describe the primary idea, motivation, and significance of the paper in a way that is accessible to any technically inclined listener.
	2.	Layer 2 – Intermediate Summary:
Delve deeper by explaining the core methodologies, key insights, and main contributions of the research. Tailor this explanation to a machine learning Ph.D.-level audience, ensuring that you articulate the underlying concepts and rationale clearly.
	3.	Layer 3 – Comprehensive Technical Walkthrough:
Deliver a fully detailed, step-by-step technical description of the problem, the proposed solution, experimental setup, results, and implications. Do not skip any detail—even the minor ones—as if you were teaching the content in an extended, narrated lecture.

Your final narrative should flow naturally as an engaging audiobook lecture, with a clear, conversational tone. By the end, your audience should feel fully equipped to explain the research to others, having understood both the big picture and all technical intricacies.

Keep the tone formal and academic. Don't use any casual language or phrases unless they are part of the paper.

The paper content is as follows:
{RESEARCH_PAPER_CONTENT}
"""

# Gemini API prompt for PDF extraction
# GEMINI_EXTRACTION_PROMPT = """
# You are an expert at extracting and summarizing content from research papers for audiobook conversion.
# Your task is to process the following research paper content and prepare it for audio narration:

# <research_paper>
# {RESEARCH_PAPER_CONTENT}
# </research_paper>

# Please follow these steps to extract and format the content:

# 1. Read through the entire paper carefully.

# 2. Extract and structure the content as follows:
#    a. Title and Authors: Present the paper's title and list of authors but not their affiliations.
#    b. Abstract: Ignore the abstract.
#    c. Main Sections: Extract the key content from the introduction, related work, methodology, results, and conclusion (if there are any other sections, add them to the result too). Omit any references, citations, and footnotes.
#    d. Mathematical Formulas: Convert equations and formulas into spoken language when possible. For complex formulas that cannot be easily verbalized, provide a brief description of their purpose or significance.
#    e. Figures and Tables: Briefly describe important visual elements, focusing on their key insights and relevance to the overall research.

# 3. Format the text for natural, engaging audio narration:
#    a. Use clear transitions between sections.
#    b. Break down long, complex sentences into more digestible parts.
#    c. Spell out acronyms and abbreviations on first use.
#    d. Assume the reader is a scientist or engineer with a strong background in the subject matter.
#    e. Use the active voice, assuming that you are narrating YOUR OWN paper to the reader.

# 4. Maintain the academic tone while optimizing for listening comprehension:
#    a. Use clear, concise language.
#    b. Emphasize key findings and their implications.
#    c. Provide context for technical terms or concepts when necessary.

# 5. Add [pause] marker where there should be a pause in the audio, especially between major sections.

# Remember to focus on creating a coherent, engaging narrative that captures the essence of the research paper in a format suitable for audio presentation.

# Aim for a total length of 15000-20000 characters.
# """  # noqa: E501

GEMINI_EXTRACTION_PROMPT = """
Imagine you are the original author of this research paper. Your task is to present an in-depth, audiobook-style explanation of your work that leaves your audience—comprising machine learning Ph.Ds and other experts—a complete understanding of every facet of your research. 

Structure your explanation in the following layers:
	1.	Layer 1 – High-Level Overview:
Provide a concise, broad summary of the research. Describe the primary idea, motivation, and significance of the paper in a way that is accessible to any technically inclined listener.
	2.	Layer 2 – Intermediate Summary:
Delve deeper by explaining the core methodologies, key insights, and main contributions of the research. Tailor this explanation to a machine learning Ph.D.-level audience, ensuring that you articulate the underlying concepts and rationale clearly.
	3.	Layer 3 – Comprehensive Technical Walkthrough:
Deliver a fully detailed, step-by-step technical description of the problem, the proposed solution, experimental setup, results, and implications. Do not skip any detail—even the minor ones—as if you were teaching the content in an extended, narrated lecture.

Your final narrative should flow naturally as an engaging audiobook lecture, with a clear, conversational tone. By the end, your audience should feel fully equipped to explain the research to others, having understood both the big picture and all technical intricacies.

Keep the tone formal and academic. Don't use any casual language or phrases unless they are part of the paper.

The paper content is as follows:
{RESEARCH_PAPER_CONTENT}
"""

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
