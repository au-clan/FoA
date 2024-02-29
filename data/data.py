import json
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class GameOf24Data:
    path = "data/datasets/24_tot.csv"
    data = pd.read_csv(path).Puzzles.tolist()

    def get_train(self):
        return self.data[0:200]

    def get_validation(self):
        return self.data[800:900]

    def get_test(self):
        return self.data[900:1000]
    
@dataclass(frozen=True)
class CrosswordsData:

    file_path = "data/datasets/mini0505.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    def get_train(self):
        return self.data[40:80]

    def get_validation(self):
        return self.data[20:40]

    def get_test(self):
        return self.data[0:20]