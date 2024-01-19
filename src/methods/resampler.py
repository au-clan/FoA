import numpy as np

class Resampler():
    def __init__(self):
        pass

    @staticmethod
    def resample_normalization(values: np.array) -> list:
        sum = values.sum()
        p = (values / sum).tolist()
        indices = list(range(len(values)))
        choices = np.random.choice(indices, size=len(values), replace=True, p=p)
        return choices.tolist()

    def resample(self, values: np.array, resample_method: str="normalization") -> list:
        if resample_method == 'normalization':
            return self.resample_normalization(values)
        else:
            raise ValueError(f"Resample method {resample_method} not implemented.")