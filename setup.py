"""Setup script for the pdf_to_audiobook package."""

from setuptools import find_packages
from setuptools import setup

if __name__ == '__main__':
  setup(
    name='pdf_to_audiobook',
    packages=find_packages(),
    include_package_data=True,
  )
