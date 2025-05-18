import sys
import os
from dataclasses import dataclass
sys.path.append(os.getcwd()) # Project root!!
from src.states.gameof24 import GameOf24State


@dataclass(frozen=True)
class RegistryEntry:
    state: GameOf24State
    verifiers: dict
    reflexions: dict