"""Test registry (§4, §5 eval/registry.py).

Every (sector, horizon, model, threshold_k) evaluated is registered BEFORE
running. The BH denominator is the registry size — you cannot fish for a
significant cell and then pretend you only ran one test.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entry:
    sector: str
    horizon: int
    model: str
    threshold_k: float


@dataclass
class Registry:
    entries: list[Entry] = field(default_factory=list)

    def register(self, sector: str, horizon: int, model: str, threshold_k: float) -> Entry:
        e = Entry(sector, horizon, model, threshold_k)
        self.entries.append(e)
        return e

    def __len__(self) -> int:
        return len(self.entries)
