"""Check the baseline interpreter and complete dependency set; emit a receipt."""
import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import re
import sys


def probe(lock):
    pins = dict(re.findall(r'^([\w.-]+)==([^\s]+)', Path(lock).read_text(), re.M))
    normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    installed = {normalize(d.metadata['Name']): d.version for d in distributions()}
    errors = []
    if platform.python_version() != '3.12.14' or platform.machine().lower() not in ('amd64', 'x86_64'):
        errors.append('Baseline requires CPython 3.12.14 x86_64')
    if platform.python_implementation() != 'CPython':
        errors.append('CPython required')
    for name, version in pins.items():
        if installed.get(normalize(name)) != version:
            errors.append(f'{name}: expected {version}, got {installed.get(normalize(name))}')
    extras = sorted(set(installed) - set(map(normalize, pins)) - {'attractive-spectrum-scalper'})
    if extras:
        errors.append('Unpinned distributions: ' + ', '.join(extras))
    if 'attractive-spectrum-scalper' in installed and installed['attractive-spectrum-scalper'] != '0.3.0':
        errors.append('Unexpected project package version')
    return dict(status='PASS' if not errors else 'FAIL', errors=errors, python=sys.version,
                executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
                platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                lock_sha256=hashlib.sha256(Path(lock).read_bytes()).hexdigest(), distributions=installed,
                numerical_environment={n: os.environ.get(n) for n in
                    ('PYTHONHASHSEED', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    receipt = probe(args.lock)
    text = json.dumps(receipt, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text, encoding='utf8', newline='\n')
    print(text)
    sys.exit(0 if receipt['status'] == 'PASS' else 1)
