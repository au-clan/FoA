from typing import List, Tuple
def linear(values: List[float]):
    """
    Compute the probability of each value.
    """
    eps = 1e-6
    values = [value + eps for value in values]
    total = sum(values)
    return [value / total for value in values]