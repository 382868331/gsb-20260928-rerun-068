"""内容寻址的数据版本标识。

版本号只依赖参与记录的内容，与文件顺序无关（预测记录按样本身份排序后
再哈希），因此同一批数据无论从哪里读入都得到相同版本。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_version(parts: Iterable[tuple[str, Any]]) -> str:
    """由若干命名部分计算 16 位内容版本（sha256 前缀）。"""
    digest = hashlib.sha256()
    for name, value in parts:
        digest.update(_canonical([name, value]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def deterministic_weights(ids: list[str], salt: str = "bootstrap") -> list[int]:
    """样本身份到稳定整数的映射，用于可复现的重抽样（不依赖抽样顺序）。"""
    out: list[int] = []
    for sample_id in ids:
        h = hashlib.sha256(f"{salt}:{sample_id}".encode("utf-8")).digest()
        out.append(int.from_bytes(h[:8], "big"))
    return out
