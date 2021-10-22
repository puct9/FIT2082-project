# Wine dataset
import numpy as np
import pandas as pd

from .bases import Dataset, get_filepath


class WineDataset(Dataset):
    """Simple convenience dataset. Call `WineDataset.load` to instantiate."""

    @classmethod
    def load(cls, test_split: float = 0.4, seed: int = None) -> "WineDataset":
        """Load the wine dataset

        Args:
            test_split: Proportion of data to be used in test split
            seed: Random seed

        Returns:
            Loaded dataset
        """
        if seed is not None:
            np.random.seed(seed)
        data = pd.read_csv(get_filepath("datasets/wine/wine_quality.csv"))
        arr = data.to_numpy()
        np.random.shuffle(arr)
        train_cutoff = int((1 - test_split) * arr.shape[0]) + 1
        return cls(
            # Train split
            arr[:train_cutoff, :-1],
            arr[:train_cutoff, -1],
            # Test (holdout) split
            arr[train_cutoff:, :-1],
            arr[train_cutoff:, -1],
        )
