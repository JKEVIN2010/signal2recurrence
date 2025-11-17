"""
Signal2Recurrence: Deep Metric Learning for Sequential Signal Analysis

A generalizable pipeline for analyzing sequential signals through recurrence plot 
transformation and deep metric learning.
"""

__version__ = "0.1.0"
__author__ = "Kevin [Your Last Name]"
__email__ = "your.email@psu.edu"

from .pipeline import SignalPipeline
from .preprocessing import SignalPreprocessor
from .recurrence import RecurrencePlotGenerator
from .siamese import SiameseNetwork

__all__ = [
    'SignalPipeline',
    'SignalPreprocessor', 
    'RecurrencePlotGenerator',
    'SiameseNetwork'
]
