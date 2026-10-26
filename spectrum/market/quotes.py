"""Public array adapter for canonical quote publications; never reconstructs a book."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuoteColumns:
    quote_t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    valid: np.ndarray
    epoch: np.ndarray

    def __post_init__(self):
        n = len(self.quote_t)
        for name in ('quote_t', 'bid', 'ask', 'valid', 'epoch'):
            array = np.asarray(getattr(self, name))
            if array.shape != (n,) or not n:
                raise ValueError('Nonempty aligned one-dimensional quote arrays required')
            expected = bool if name == 'valid' else np.int64
            if array.dtype != expected:
                raise TypeError(f'{name} must have dtype {np.dtype(expected)}')
        if np.any(np.diff(self.quote_t) < 0):
            raise ValueError('Publication order regressed; equal timestamps retain row order')
        if np.any(self.valid & ((self.bid <= 0) | (self.ask < self.bid) | (self.epoch < 0))):
            raise ValueError('Valid quotes require supported positive, uncrossed BBO')

    @classmethod
    def from_frame(cls, frame):
        """Canonical FINALIZED_MARKET_V2 columns; available_ns is the read clock."""
        return cls(*(frame[name].to_numpy(copy=True) for name in
                     ('available_ns', 'bid_nano', 'ask_nano', 'valid', 'epoch')))
