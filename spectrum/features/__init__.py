"""Feature families share source clocks; frame and event inputs remain explicit."""
from .contract import Context, FeaturePlugin, FeatureBatch
from .pipeline import compute, compute_batch, generate, register, load_plugins
from .stream import FeatureStream
from .catalog import families as list_families, get_family
__all__ = ['Context', 'FeaturePlugin', 'FeatureBatch', 'compute', 'compute_batch', 'generate', 'register', 'load_plugins', 'FeatureStream', 'list_families', 'get_family']
