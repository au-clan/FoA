from src.verifiers.Verifier import Verifier
from src.states.gameof24 import GameOf24State
from typing import Any

class TextVerifier(Verifier):
    """Just a random type of verifier."""
    def __init__(self):
        self.name = "Text Verifier"
    
    @staticmethod
    def check_all(state: GameOf24State, last_step) -> Any:
        return True, "The whatever verification"