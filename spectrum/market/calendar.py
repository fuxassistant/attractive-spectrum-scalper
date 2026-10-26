"""The shared venue schedule from the canon; feed coverage is a separate mask."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tzdata
import spectrum.canon as canon_module
from spectrum.canon import load_canon


@dataclass(frozen=True)
class MarketCalendar:
    canon_name: str = 'canon_v1'
    venue: str = 'gc_globex'

    def __post_init__(self):
        if self.venue not in load_canon(self.canon_name).venues:
            raise ValueError('Unknown canon venue: ' + self.venue)

    def closed(self, timestamp_ns):
        return ~load_canon(self.canon_name).venues[self.venue].contains(timestamp_ns)

    def contract(self):
        return dict(version='CANON_CALENDAR_V1', canon_name=self.canon_name, venue=self.venue,
                    canon_fingerprint=load_canon(self.canon_name).fingerprint,
                    canon_implementation_sha256=hashlib.sha256(Path(canon_module.__file__).read_bytes()).hexdigest(),
                    tzdata_version=tzdata.__version__)

    @classmethod
    def restore(cls, contract):
        if contract.get('version') != 'CANON_CALENDAR_V1':
            raise ValueError('Unsupported calendar contract; regenerate legacy feature checkpoints')
        calendar = cls(contract['canon_name'], contract['venue'])
        if calendar.contract() != contract:
            raise ValueError('Feature calendar drift: canon or timezone rules changed')
        return calendar
