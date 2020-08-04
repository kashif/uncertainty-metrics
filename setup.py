"""Uncertainty Metrics setup.py."""

import os
import sys

from setuptools import find_packages
from setuptools import setup

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'uncertainty_metrics')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setup(
    name='uncertainty_metrics',
    version=__version__,
    description='Uncertainty Metrics',
    author='Uncertainty Metrics Team',
    author_email='jeremynixon@google.com',
    url='http://github.com/google/uncertainty_metrics',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=['numpy>=1.7'],
    extras_require={
        'numpy': ['matplotlib>=2.0.0',
                  'scipy>=1.0.0',
                  'scikit-learn>=0.20.0'],
        'tensorflow': ['tensorflow>=2.0.0',
                       'tensorflow_probability>=0.9'],
        'tf-nightly': ['tf-nightly',
                       'matplotlib>=2.0.0',
                       'scipy>=1.0.0',
                       'scikit-learn>=0.20.0'
                       'tensorflow>=2.0.0',
                       'tensorflow_probability>=0.9'],
        'tests': [
            'absl-py>=0.5.0',
            'pylint>=1.9.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='probabilistic programming tensorflow machine learning',
)
