import random

from typing import List
from scipy.special import softmax
import numpy as np

from utils import sigmoid


class Resampler: 
    def __init__(self, randomness: int):
        self.randomness = randomness 

    def resample(self, r, n_picks, resampling_method, include_init=True):
        methods = {
            "linear": Resampler.linear,
            "logistic": Resampler.logistic,
            "max": Resampler.max,
            "percentile": Resampler.percentile
        }

        if resampling_method not in methods:
            raise ValueError(f"Invalid resampling method: {resampling_method}\nValid methods: {methods.keys()}")
        
        if n_picks == 0:
            return [], []
        
        if not include_init:
            r = r.copy()
            r["values"] = r["values"][1:]
            r["states"] = r["states"][1:]
            r["idx"] = r["idx"][1:]

        # Get probabilities for each state based on values
        probabilities = methods[resampling_method](r["values"])
        resampled_indices = np.random.choice(range(len(r["values"])), size=n_picks, p=probabilities, replace=True).tolist()
        
        # Resample states based on resampled_indices
        random.seed(self.randomness)
        new_randomness = [random.randint(1, 1000) for _ in range(n_picks)]
        self.randomness = new_randomness[-1]
        resampled_states = [r["states"][i].duplicate(randomness) for i, randomness in zip(resampled_indices, new_randomness)]
        return resampled_states, resampled_indices
    
    @staticmethod
    def linear(values: List[float])-> List[float]:
        """
        Compute the linear probability of each value.
        """
        eps = 1e-6
        values = [value + eps for value in values]
        total = sum(values)
        return [value / total for value in values]
    
    @staticmethod
    def logistic(values: List[float])-> List[float]:
        """
        Computes the logistic probability of each value.
        """
        values = [sigmoid(value) for value in values]
        return softmax(values)

    @staticmethod
    def max(values: List[float])-> List[float]:
        """
        Computes uniform probability of highest values solely.
        """
        max_value = max(values)
        values = [value if value==max_value else 0 for value in values]
        return Resampler.linear(values)
    
    @staticmethod
    def percentile(values: List[float], percentile: float=0.75) -> List[float]:
        """
        Computes the linear probability considering only the highest percentile values.
        """
        threshold = np.percentile(values, percentile)
        values = [value if value >= threshold else 0 for value in values]
        return Resampler.linear(values)