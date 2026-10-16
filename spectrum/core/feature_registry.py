"""Small, explicit extension contract: named fields over a causal context."""
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    unit: str = 'dimensionless'
    meaning: str = ''


@dataclass(frozen=True)
class Family:
    name: str
    version: str
    fields: tuple[Field, ...]
    compute: Callable[[object], Mapping[str, float]]


class Registry:
    """No global registration or dynamic discovery of columns during inference."""
    def __init__(self, families):
        self.families = tuple(families)
        self.names, self.indices, self.specs = [], {}, []
        for family in self.families:
            if not family.name or not family.version or family.name in self.indices:
                raise ValueError('Feature families need unique names and explicit versions')
            self.indices[family.name] = []
            for field in family.fields:
                if not field.name or field.name.endswith('_known') or field.name in self.names:
                    raise ValueError('Duplicate or reserved feature name: ' + field.name)
                self.indices[family.name].extend((len(self.names), len(self.names) + 1))
                self.names.extend((field.name, field.name + '_known'))
                self.specs.append(dict(name=field.name, unit=field.unit, meaning=field.meaning,
                                       family=family.name, version=family.version))
        self.contract = [dict(name=f.name, version=f.version,
                              fields=[dict(name=x.name, unit=x.unit, meaning=x.meaning) for x in f.fields])
                         for f in self.families]

    def compute(self, context):
        row = np.zeros(len(self.names), np.float32)
        position = 0
        for family in self.families:
            result = family.compute(context)
            if set(result) != {field.name for field in family.fields}:
                raise ValueError('Feature output differs from schema: ' + family.name)
            for field in family.fields:
                value = float(result[field.name])
                if np.isfinite(value):
                    if abs(value) > np.finfo(np.float32).max:
                        raise ValueError('Feature overflows float32: ' + field.name)
                    row[position:position + 2] = value, 1.
                position += 2
        return row
