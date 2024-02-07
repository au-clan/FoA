import numpy as np

class Resampler():
    def __init__(self):
        pass

    @staticmethod
    def resample_normalization(values: np.array) -> list:
        """
        Normalized resampling.
        """
        sum = values.sum()
        p = (values / sum).tolist()
        indices = list(range(len(values)))
        choices = np.random.choice(indices, size=len(values), replace=True, p=p)
        return choices.tolist()

    def resample_greedy(self, values: np.array) -> list:
        """
        Normalized resampling considering only highest values.
        
        Eg. for np.array([2,2,1,0]) only indices 0 and 1 will be considered.
        """
        mask = values != values.max()
        values[mask]=0
        return self.resample_normalization(values)

    def resample(self, values: np.array, resample_method: str="normalization") -> list:
        if resample_method == 'normalization':
            return self.resample_normalization(values)
        elif resample_method == "greedy":
            return self.resample_greedy(values)
        else:
            raise ValueError(f"Resample method {resample_method} not implemented.")