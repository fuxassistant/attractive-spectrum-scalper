from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json

NS=10**9
PLANS=((12,8,120),(24,12,300),(40,20,600))

@dataclass(frozen=True)
class Contract:
    version: str='LSTM_PLAYER_V1'
    size_oz: float=1.
    fee_roundtrip: float=.30
    initial_capital: float=1000.
    latency_ns: int=100_000_000
    pending_timeout_ns: int=2*NS
    support_gap_ns: int=5*NS
    daily_drawdown_stop: float=30.
    london_open_hour: int=8
    new_york_close_hour: int=17
    terminal_reserve_seconds: int=30
    hidden_size: int=48
    epochs: int=30
    ppo_passes: int=3
    learning_rate: float=.0003
    reward_scale: float=5.
    gamma: float=1.
    gae_lambda: float=.95
    clip_ratio: float=.2
    entropy_weight: float=.01
    gradient_clip: float=.5
    seed: int=61015

    def dict(self):return asdict(self)

def code_identity():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
