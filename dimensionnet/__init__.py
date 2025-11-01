"""
DimensionNet: Unveiling Hidden Realities Through Deep Learning

A research-grade framework for detecting and analyzing higher-dimensional
structures in data using deep learning and topological methods.
"""

__version__ = "1.0.0"
__author__ = "Mayank Singh"
__email__ = "mayanksiingh2@gmail.com"

from .models.vae import VAE
from .data.datasets import load_sample_data

__all__ = ['VAE', 'load_sample_data']
