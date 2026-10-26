"""Reusable formulas extracted from spectrum/ingest.py; see docs/MIGRATION.json."""
from pathlib import Path
import json
from spectrum.core.storage import sha

def save_json(path, value):
    path=Path(path); temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)
