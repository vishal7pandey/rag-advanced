from __future__ import annotations
from typing import List
from app.core.types import Turn


class WindowMemory:
    def __init__(self) -> None:
        self._turns: List[Turn] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))

    def get(self, n: int) -> List[Turn]:
        if n <= 0:
            return []
        return self._turns[-n:]

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)
