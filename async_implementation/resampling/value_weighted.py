from typing import List
from scipy.special import softmax
import numpy as np

from utils import sigmoid

def linear(values: List[float])-> List[float]:
    """
    Compute the linear probability of each value.
    """
    eps = 1e-6
    values = [value + eps for value in values]
    total = sum(values)
    return [value / total for value in values]

def logistic(values: List[float])-> List[float]:
    """
    Computes the logistic probability of each value.
    """
    values = [sigmoid(value) for value in values]
    return softmax(values)

def max(values: List[float])-> List[float]:
    """
    Computes uniform probability of highest values solely.
    """
    max_value = max(values)
    values = [value if value==max_value else 0 for value in values]
    return linear(values)

def percentile(values: List[float], percentile: float=0.75) -> List[float]:
    """
    Computes the linear probability considering only the highest percentile values.
    """
    threshold = np.percentile(values, percentile)
    values = [value if value >= threshold else 0 for value in values]
    return linear(values)