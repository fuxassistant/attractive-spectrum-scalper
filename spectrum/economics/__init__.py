"""Economic layer: the only place where costs and money enter.

Reads facts from spectrum.future and spectrum.crossings; never writes them.
"""
from .marks import future_crossing_marks
__all__ = ['future_crossing_marks']
