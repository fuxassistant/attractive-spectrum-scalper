"""Future facts: everything that is known only after the origin.

Each public function and every output field here is named with "future" and
returns marked results (spectrum.leakage: name, FutureFrame type, value
fingerprints). Model input rejects all three; future_examples is the only
join of X and Y, by publication time. Costs and money: spectrum.economics.
"""
from .crossings import future_crossings, future_support_end, FUTURE_STATUSES
from .examples import future_examples
__all__ = ['future_crossings', 'future_support_end', 'FUTURE_STATUSES', 'future_examples']
