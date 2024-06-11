import json
import pandas as pd
from dataclasses import dataclass

from utils import get_file_names

@dataclass(frozen=True)
class GameOf24Data:
    path = "data/datasets/24_tot.csv"
    data = pd.read_csv(path).Puzzles.tolist()

    def get_data(self, set):
        if set == "mini":
            indices = list(range(0,10))
        elif set == "train":
            indices = list(range(850,875)) + list(range(1025,1050))
        elif set == "validation":
            indices = list(range(875,900)) + list(range(1050,1075))
        elif set == "test":
            indices = list(range(900,1000))
        else:
            raise ValueError("Invalid set name")

        return indices, [self.data[i] for i in indices]
    
@dataclass(frozen=True)
class CrosswordsData:

    file_path = "data/datasets/mini0505.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    def get_data(self, set):
        if set == "mini":
            indices = [i + 1 for i in range(0,100,5)[:5]]
        elif set == "train":
            indices = [i + 2 for i in range(0,100,5)[:10]]
        elif set == "validation":
            indices = [i + 3 for i in range(0,100,5)]
        elif set == "test":
            indices = [i for i in range(0,100,5)]
        else:
            raise ValueError("Invalid set name")
        
        return indices, [self.data[i] for i in indices]

@dataclass(frozen=True)
class TextWorldData:
        
    data = sorted([f for f in get_file_names("data/datasets/tw_games") if f.endswith(".ulx")])


    def get_data(self, set, challenge="cooking", level=None):
        
        wrong_challenge_message = "Avaialble challenges are: simple, cooking, collector, hunter"
        assert challenge in ["simple", "cooking", "coin", "treasure"], wrong_challenge_message
        data = [d for d in self.data if d.startswith(f"tw-{challenge}_")]

        if level is not None:
            data = [d for d in data if f"level_{level}" in d]

        if challenge=="simple":
            data = [d for d in data if f"rewards_balanced" in d]
        
        if set == "mini":
            indices = list(range(5))
        elif set == "train":
            indices = list(range(100))
        else:
            raise ValueError("Invalid set name")
        return indices, [data[i] for i in indices]
    
@dataclass(frozen=True)
class WebShopData:
    """
    Data is actually handled by the server so we only need to select the indices
    """
    
    def get_data(self, set):

        if set == "mini":
            indices = list(range(5))
        elif set == "train":
            indices = list(range(70, 85))
        elif set == "validation":
            indices = list(range(55, 70))
        elif set == "test":
            indices = list(range(5, 55))
        elif set == "test1":
            indices = list(range(5, 15))
        elif set == "test2":
            indices = list(range(15, 25))
        elif set == "test3":
            indices = list(range(25, 35))
        elif set == "test4":
            indices = list(range(35, 45))
        elif set == "test5":
            indices = list(range(45, 55))
        else:
            raise ValueError("Invalid set name")
        
        return indices, indices


