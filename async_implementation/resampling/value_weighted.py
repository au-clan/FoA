from typing import List
from utils import sigmoid
from scipy.special import softmax



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

