from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GameOf24State:
    # game of 24 puzzle, for example 1 1 4 6
    puzzle: str

    # initialized to the same value as puzzle, but is updated as the game progresses
    current_state: str

    steps: List[str]

    randomness: int

    def __hash__(self):
        return hash((self.puzzle, self.current_state, " -> ".join(self.steps)))
    
    def items(self):
        return self.puzzle, self.current_state, self.steps, self.randomness