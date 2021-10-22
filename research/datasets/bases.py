import glob
import os
from typing import Dict, Generator, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import KFold

BASE_DIR = os.environ.get("root_dir", ".")


def get_filepath(relative: str):
    return os.path.join(BASE_DIR, relative)


class Dataset:
    """Base class for datasets

    Args:
        x_train: Training inputs
        y_train: Training targets
        x_test: Testing inputs
        y_test: Testing targets

    Attributes:
        x_train: Training inputs
        y_train: Training targets
        x_test: Testing inputs
        y_test: Testing targets
        x_all: Combined training and testing inputs
        y_all: Combined training and testing targets
    """

    @classmethod
    def load(cls) -> Union["Dataset", List["Dataset"]]:
        raise NotImplementedError

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        # Test is holdout
        # Shapes
        # x_train (N, X)
        # y_train (N,)
        # x_test  (M, X)
        # y_test  (M,)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_all: np.ndarray = np.concatenate((x_train, x_test))
        self.y_all: np.ndarray = np.concatenate((y_train, y_test))

    def generate_folds(
        self, n_splits: int = 10
    ) -> Generator[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None
    ]:
        """Generates folds for k-fold cross-validation

        Args:
            n_splits: Number of splits (parameter k of k-fold) cross-validation

        Yields:
            Iterable of folds, each item being a 4-tuple of
            (x_train, y_train, x_val, y_val)
        """
        kf = KFold(n_splits=n_splits)
        for train, val in kf.split(self.x_train):
            # train and val are arrays of indices
            # Yield (x_train, y_train, x_val, y_val)
            yield (
                self.x_train[train],
                self.y_train[train],
                self.x_train[val],
                self.y_train[val],
            )


class AutoLoadedDataset(Dataset):
    """
    Automatically lodead datasets. Do not call constructor directly. Instead,
    call `AutoLoadedDataset.load`.

    Args:
        x_train: Training inputs
        y_train: Training targets
        x_test: Testing inputs
        y_test: Testing targets
        name: Name of dataset

    Attributes:
        x_train: Training inputs
        y_train: Training targets
        x_test: Testing inputs
        y_test: Testing targets
        x_all: Combined training and testing inputs
        y_all: Combined training and testing targets
        name: Name of dataset
    """

    @staticmethod
    def load_arff(fname: str) -> pd.DataFrame:
        return pd.DataFrame(arff.loadarff(fname)[0])

    @classmethod
    def load(
        cls,
        test_split: float = 0.4,
        seed: int = None,
    ) -> Dict[str, "AutoLoadedDataset"]:
        """Load all datasets

        Args:
            test_split: Proportion of data to be used in test split
            seed: Random seed

        Returns:
            Dict mapping dataset name to `AutoLoadedDataset` object
        """
        directory = get_filepath("datasets/auto")
        dataframes = {}

        # Search for .csv files
        files = glob.glob(directory + "/**/*.csv", recursive=True)
        for file in files:
            data = pd.read_csv(file)
            dataframes[file] = data

        # Search for .arff files
        files = glob.glob(directory + "/**/*.arff", recursive=True)
        for file in files:
            try:
                data = cls.load_arff(file)
                assert data.shape[1] > 1
            except (NotImplementedError, ValueError, AssertionError):
                print(f"Unable to load {file}")
                continue
            dataframes[file] = data

        loaded = {}
        for file, data in dataframes.items():
            if seed is not None:
                np.random.seed(seed)
            arr = data.to_numpy()
            np.random.shuffle(arr)
            train_cutoff = int((1 - test_split) * arr.shape[0]) + 1
            fname = ".".join(os.path.split(file)[1].split(".")[:-1])
            loaded[fname] = cls(
                # Train split
                arr[:train_cutoff, :-1],
                arr[:train_cutoff, -1],
                # Test (holdout) split
                arr[train_cutoff:, :-1],
                arr[train_cutoff:, -1],
                fname,
            )
        print(f"Loaded {len(loaded)} datasets")
        return loaded

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        name: str,
    ) -> None:
        super().__init__(x_train, y_train, x_test, y_test)
        self.name = name
