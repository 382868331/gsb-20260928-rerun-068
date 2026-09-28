from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .versions import canonical_hash


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    y_true: str
    weight: float = 1.0
    groups: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.y_true:
            raise ValueError("y_true must be non-empty")
        weight = float(self.weight)
        if weight <= 0:
            raise ValueError("weight must be positive")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "groups", dict(self.groups))


class SampleSet:
    """Immutable, versioned collection of evaluation samples."""

    def __init__(self, samples: Iterable[Sample | Mapping[str, Any]]):
        records = [item if isinstance(item, Sample) else Sample(**item) for item in samples]
        seen: set[str] = set()
        for sample in records:
            if sample.sample_id in seen:
                raise ValueError(f"duplicate sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
        self._samples = tuple(records)
        self._by_id = {sample.sample_id: sample for sample in self._samples}
        self._classes = tuple(dict.fromkeys(sample.y_true for sample in self._samples))
        self._group_names = tuple(
            dict.fromkeys(name for sample in self._samples for name in sample.groups)
        )
        self._version = canonical_hash(
            [
                {
                    "sample_id": sample.sample_id,
                    "y_true": sample.y_true,
                    "weight": sample.weight,
                    "groups": sample.groups,
                }
                for sample in self._samples
            ]
        )

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self._samples

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self._samples)

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    @property
    def group_names(self) -> tuple[str, ...]:
        return self._group_names

    @property
    def version(self) -> str:
        return self._version

    def get(self, sample_id: str) -> Sample | None:
        return self._by_id.get(sample_id)

    def group_value(self, sample_id: str, group_name: str) -> str | None:
        sample = self._by_id.get(sample_id)
        if sample is None:
            return None
        return sample.groups.get(group_name)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "SampleSet":
        normalized = []
        for record in records:
            item = dict(record)
            groups = item.pop("groups", {})
            for key, value in list(item.items()):
                if key.startswith("group:"):
                    groups[key.removeprefix("group:")] = value
                    item.pop(key)
            item.setdefault("groups", groups)
            normalized.append(item)
        return cls(normalized)

    @classmethod
    def from_json(cls, path: str | Path) -> "SampleSet":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_records(payload)

    @classmethod
    def from_csv(cls, path: str | Path) -> "SampleSet":
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        records = []
        for row in rows:
            item = {key: value for key, value in row.items() if value is not None}
            if "weight" in item:
                item["weight"] = float(item["weight"])
            records.append(item)
        return cls.from_records(records)
