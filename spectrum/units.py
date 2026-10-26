"""Named units for the library; values are declared in the canon (resources/canon_v1.json).

Protected sources keep their own literals; their meaning is annotated in
canon_v1.json legacy_conventions.units and checked by tests/test_measures.py.
"""
from spectrum.resources import contract

_DECLARED = contract('canon_v1')['constants']

NS = int(_DECLARED['ns_per_second']['value'])
MINUTE = 60 * NS
HOUR = 60 * MINUTE
DAY = 24 * HOUR
PRICE_NANO_PER_USD = int(_DECLARED['price_nano_per_usd']['value'])
GC_TICK_USD = float(_DECLARED['gc_tick_usd']['value'])
GC_TICK_NANO = int(_DECLARED['gc_tick_nano']['value'])
GC_CONTRACT_OZ = int(_DECLARED['gc_contract_oz']['value'])

if GC_TICK_NANO != round(GC_TICK_USD * PRICE_NANO_PER_USD):
    raise ValueError('Canon GC tick in USD and nano-USD disagree')
