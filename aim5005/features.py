import numpy as np
from typing import List, Tuple

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a NumPy array or a list"
        return x
    
    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        diff_max_min[diff_max_min == 0] = 1  # Avoid division by zero
        return (x - self.minimum) / diff_max_min   #Modified 
    
    def fit_transform(self, x: list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None  # Standard deviation
    
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure the input is a NumPy array.
        """
        #Modified 
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a NumPy array or a list"
        return x

    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=0)  # ddof=0 ensures population standard deviation

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        std_adj = np.where(self.std == 0, 1, self.std)  # Prevent division by zero
        return (x - self.mean) / std_adj

    def fit_transform(self, x: list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)


#Implemented Label Encoder
class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        """Fit the LabelEncoder to the provided labels."""
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """Transform the labels to integer values."""
        if self.classes_ is None:
            raise ValueError("Encoder not fitted yet.")
        mapping = {label: index for index, label in enumerate(self.classes_)}
        return np.array([mapping[label] for label in y])

    def fit_transform(self, y):
        """Fit the encoder and then transform the labels."""
        self.fit(y)
        return self.transform(y)
