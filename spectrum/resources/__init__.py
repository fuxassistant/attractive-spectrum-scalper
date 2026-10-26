"""Versioned semantic contracts shipped with the library."""
import json
from importlib.resources import files

def contract(name: str) -> dict:
    if not name.replace('_', '').isalnum():
        raise ValueError('A contract name, not a filesystem path, is required')
    return json.loads(files(__package__).joinpath(name + '.json').read_text(encoding='utf-8'))
