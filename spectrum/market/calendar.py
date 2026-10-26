"""Frozen observed-hours convention, separate from unexpected source loss.

May 2026 OHLC evidence: daily 21:00--22:00 UTC, Fri 21:00--Sun 22:00.
This is a bounded research calendar, not a universal holiday calendar.
"""
from dataclasses import dataclass, asdict
import numpy as np

NS=10**9

@dataclass(frozen=True)
class MarketCalendar:
    break_start_utc: int = 21
    break_hours: int = 1
    version: str = 'OBSERVED_MAY_2026_V1'

    def closed(self, timestamp_ns):
        t=np.asarray(timestamp_ns,dtype=np.int64)
        day=t//(86400*NS);hour=(t//(3600*NS))%24; dow=(day+3)%7
        end=self.break_start_utc+self.break_hours
        return ((dow==5)|((dow==4)&(hour>=self.break_start_utc))|
                ((dow==6)&(hour<end))|
                ((dow<4)&(hour>=self.break_start_utc)&(hour<end)))

    def contract(self):return asdict(self)
