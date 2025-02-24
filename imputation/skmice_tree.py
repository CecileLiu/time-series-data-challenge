'''Fit for scikit-learn models
RandomForestRegressor
GradientBoostingRegressor

'''
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import Any, Tuple


class SKTreeMiceImputer:
    def __init__(self, missing_values: Any = np.nan, strategy: str = "mean", verbose: int = 0, copy: bool = True) -> None:
        self.missing_values = missing_values
        self.strategy = strategy
        self.verbose = verbose
        self.copy = copy
        self.imp = SimpleImputer(missing_values=self.missing_values, strategy=self.strategy)

    def _seed_values(self, X: np.ndarray) -> np.ndarray:
        """Impute initial missing values in the dataset."""
        return self.imp.fit_transform(X)

    def _get_mask(self, X: np.ndarray) -> np.ndarray:
        """Create a mask for missing values in the dataset."""
        return np.isnan(X) if np.isnan(self.missing_values) else X == self.missing_values

    def _process(self, X: np.ndarray, column: int, model_class: Any) -> float:
        """Impute missing values for a specific column using a tree-based model."""
        mask = self._get_mask(X)[:, column]

        known_indices = np.where(~mask)[0]  # Rows where the column is not missing
        missing_indices = np.where(mask)[0]  # Rows where the column is missing

        if len(missing_indices) == 0:
            return 1.0  # No missing values to impute

        # Prepare training data
        y_train = X[known_indices, column]
        X_train = np.delete(X[known_indices], column, axis=1)

        # Prepare test data (missing values)
        X_test = np.delete(X[missing_indices], column, axis=1)

        # Train the model
        model = model_class(n_estimators=100, random_state=42)  # Tree-based models need n_estimators
        model.fit(X_train, y_train)

        # Predict missing values
        y_pred = model.predict(X_test)

        # Replace missing values in the original array
        X[missing_indices, column] = y_pred

        return model.score(X_train, y_train)  # Model performance score

    def transform(self, X: np.ndarray, model_class: Any = RandomForestRegressor, iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Perform multiple imputation on the dataset using tree-based models."""
        X = np.array(X, dtype=float)  # Ensure X is a NumPy array
        mask = self._get_mask(X)

        # Seed initial values for imputation
        seeded_X = self._seed_values(X.copy())

        scores = np.zeros((iterations, X.shape[1]))

        for i in range(iterations):
            for c in range(X.shape[1]):
                scores[i, c] = self._process(seeded_X, c, model_class)

        return seeded_X, scores
