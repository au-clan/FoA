from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GameOf24State:
    # initialized to the same value as puzzle, but is updated as the game progresses
    current_state: str

    steps: List[str] = None
