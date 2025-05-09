from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from src.states.gameof24 import GameOf24State

class Verifier(ABC):
    def __init__(self):
        self.name = "Abstract Verifier (Interface)"
        
    @staticmethod
    @abstractmethod
    def check_all(state: GameOf24State, last_step) -> Any:
        pass