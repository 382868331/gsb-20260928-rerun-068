"""样本集：样本身份、真实标签、权重与分组属性。

样本身份（sample_id）在评估期间保持稳定，分组信息挂在样本上，后续新增
指标可以直接复用而不必重新关联数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .versioning import content_version


@dataclass(frozen=True)
class Sample:
    sample_id: str
    label: str
    weight: float = 1.0
    groups: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id 不能为空")
        if self.label is None or self.label == "":
            raise ValueError(f"样本 {self.sample_id} 缺少真实标签")
        if self.weight < 0:
            raise ValueError(f"样本 {self.sample_id} 权重不能为负")


class SampleSet:
    def __init__(self, samples: list[Sample]):
        seen: set[str] = set()
        duplicates: list[str] = []
        for sample in samples:
            if sample.sample_id in seen:
                duplicates.append(sample.sample_id)
            seen.add(sample.sample_id)
        if duplicates:
            raise ValueError(f"样本集存在重复身份: {sorted(set(duplicates))}")
        self.samples: tuple[Sample, ...] = tuple(samples)
        self._index = {s.sample_id: s for s in self.samples}
        self.version = content_version(
            [
                (
                    "samples",
                    [
                        [s.sample_id, s.label, s.weight, sorted(s.groups.items())]
                        for s in sorted(self.samples, key=lambda x: x.sample_id)
                    ],
                )
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._index

    def get(self, sample_id: str) -> Sample:
        return self._index[sample_id]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.sample_id for s in self.samples)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({s.label for s in self.samples}))
