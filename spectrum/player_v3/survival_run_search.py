"""Reproducible series-risk search on the exposed six-week development period."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
import argparse
import json
import time
from spectrum.player.contract import Contract
from spectrum.player.data import Day
from spectrum.player_v3.martingale import search_population, evaluate_frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/player_v3/martingale_series_001')
    parser.add_argument('--data', default='data/research/player_v3/memory_001')
    args = parser.parse_args()
    data = Path(args.data)
    verified = json.loads((data / 'verification.json').read_text())
    print(json.dumps({'data_verification': verified}), flush=True)
    dates = sorted(p.name for p in data.iterdir() if p.is_dir() and p.name.startswith('2026-'))
    # Source hashes were independently checked in the prepared-data receipt.
    train = [Day(data / d, verify=False) for d in dates if '2026-04-13' <= d <= '2026-05-08']
    val = [Day(data / d, verify=False) for d in dates if '2026-05-11' <= d <= '2026-05-15']
    groups = [train[i:i + 5] for i in range(0, len(train), 5)]
    c = Contract(initial_capital=200.)
    output = Path(args.output)
    started = time.perf_counter()
    frozen = search_population(train, val, output, c, training_groups=groups)
    print(json.dumps({'martingale_frozen': frozen, 'seconds': time.perf_counter() - started}), flush=True)
    # Evaluation is not instantiated until the candidate selection is frozen.
    test = [Day(data / d, verify=False) for d in dates if '2026-05-18' <= d <= '2026-05-22']
    summary = evaluate_frozen(test, frozen, output / 'evaluation', c)
    print(json.dumps({'martingale_evaluation': summary}), flush=True)


if __name__ == '__main__':
    main()
