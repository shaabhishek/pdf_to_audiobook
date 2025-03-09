"""Setup script for the pdf_to_audiobook package."""

from setuptools import find_packages
from setuptools import setup

if __name__ == '__main__':
  setup(
    name='pdf_to_audiobook',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
      'console_scripts': [
        'pdf-to-audiobook=pdf_to_audiobook.main:main',
        'pdf-to-audiobook-web=pdf_to_audiobook.web_gui:main',
      ],
    },
  )
