from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GameOf24State:
    # game of 24 puzzle, for example 1 1 4 6
    puzzle: str

    # initialized to the same value as puzzle, but is updated as the game progresses
    current_state: str

    randomness: int

    steps: List[str]

