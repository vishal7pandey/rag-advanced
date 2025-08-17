from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Protocol, Iterator, Tuple, Any

from app.core.types import AnswerBundle


class Flow(Protocol):
    def run(self, question: str, session_state: Dict) -> AnswerBundle: ...
    def run_stream(
        self, question: str, session_state: Dict
    ) -> Tuple[Iterator[str], Dict[str, Any]]: ...


@dataclass
class StepTimings:
    retrieve: float = 0.0
    rerank: float = 0.0
    generate: float = 0.0
