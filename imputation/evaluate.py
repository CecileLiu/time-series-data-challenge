import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

class MiceImpEvaluator:
    def __init__(self, original: np.ndarray, imputed: np.ndarray, missing_mask: np.ndarray, categorical_indices=None):
        """
        Initialize the evaluator with original and imputed datasets.

        Parameters:
        - original (np.ndarray): The ground-truth dataset before introducing missing values.
        - imputed (np.ndarray): The dataset after imputation.
        - missing_mask (np.ndarray): A boolean mask indicating which values were originally missing.
        - categorical_indices (list, optional): Indices of categorical columns.
        """
        self.original = np.array(original)
        self.imputed = np.array(imputed)
        self.missing_mask = np.array(missing_mask, dtype=bool)
        self.categorical_indices = set(categorical_indices) if categorical_indices else set()
        
        self._validate_inputs()

    def _validate_inputs(self):
        """Ensure input arrays have the same shape and valid data types."""
        if self.original.shape != self.imputed.shape:
            raise ValueError("Original and imputed datasets must have the same shape.")
        if self.original.shape != self.missing_mask.shape:
            raise ValueError("Missing mask must have the same shape as datasets.")

    def calculate_mse(self) -> float:
        """Compute Mean Squared Error (MSE) for numerical variables."""
        numerical_mask = ~np.isin(range(self.original.shape[1]), list(self.categorical_indices))
        missing_numerical = self.missing_mask[:, numerical_mask]
        
        return mean_squared_error(
            self.original[missing_numerical],
            self.imputed[missing_numerical]
        )

    def calculate_mae(self) -> float:
        """Compute Mean Absolute Error (MAE) for numerical variables."""
        numerical_mask = ~np.isin(range(self.original.shape[1]), list(self.categorical_indices))
        missing_numerical = self.missing_mask[:, numerical_mask]

        return mean_absolute_error(
            self.original[missing_numerical],
            self.imputed[missing_numerical]
        )

    def calculate_nrmse(self) -> float:
        """Compute Normalized Root Mean Squared Error (NRMSE) for numerical variables."""
        mse = self.calculate_mse()
        std = np.std(self.original[self.missing_mask])
        return np.sqrt(mse) / std if std != 0 else 0

    def calculate_categorical_accuracy(self) -> float:
        """Compute accuracy for categorical variable imputation."""
        if not self.categorical_indices:
            raise ValueError("No categorical variables specified for evaluation.")

        total_correct = 0
        total_missing = 0

        for col in self.categorical_indices:
            missing_cat = self.missing_mask[:, col]
            total_correct += np.sum(self.original[missing_cat] == self.imputed[missing_cat])
            total_missing += np.sum(missing_cat)

        return total_correct / total_missing if total_missing > 0 else 0

    def evaluate_all(self) -> dict:
        """Compute all evaluation metrics."""
        results = {
            "MSE": self.calculate_mse(),
            "MAE": self.calculate_mae(),
            "NRMSE": self.calculate_nrmse()
        }

        if self.categorical_indices:
            results["Categorical Accuracy"] = self.calculate_categorical_accuracy()

        return results

    def plot_numerical_distributions(self):
        """Plot original vs. imputed value distributions for numerical variables."""
        numerical_cols = list(set(range(self.original.shape[1])) - self.categorical_indices)
        num_plots = min(len(numerical_cols), 4)  # Limit to 4 plots for readability
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

        if num_plots == 1:
            axes = [axes]  # Ensure axes is iterable

        for i, col in enumerate(numerical_cols[:num_plots]):
            sns.histplot(self.original[self.missing_mask[:, col], col], kde=True, color='blue', label="Original", ax=axes[i])
            sns.histplot(self.imputed[self.missing_mask[:, col], col], kde=True, color='red', label="Imputed", ax=axes[i])
            axes[i].set_title(f"Column {col}")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_categorical_distributions(self):
        """Plot original vs. imputed category distributions for categorical variables."""
        if not self.categorical_indices:
            raise ValueError("No categorical variables specified for plotting.")

        num_plots = min(len(self.categorical_indices), 4)  # Limit to 4 plots
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

        if num_plots == 1:
            axes = [axes]

        for i, col in enumerate(list(self.categorical_indices)[:num_plots]):
            original_counts = np.unique(self.original[self.missing_mask[:, col], col], return_counts=True)
            imputed_counts = np.unique(self.imputed[self.missing_mask[:, col], col], return_counts=True)

            original_dict = dict(zip(original_counts[0], original_counts[1]))
            imputed_dict = dict(zip(imputed_counts[0], imputed_counts[1]))

            all_categories = sorted(set(original_dict.keys()) | set(imputed_dict.keys()))
            original_values = [original_dict.get(cat, 0) for cat in all_categories]
            imputed_values = [imputed_dict.get(cat, 0) for cat in all_categories]

            width = 0.4
            x = np.arange(len(all_categories))

            axes[i].bar(x - width/2, original_values, width=width, label="Original", color='blue')
            axes[i].bar(x + width/2, imputed_values, width=width, label="Imputed", color='red')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(all_categories, rotation=45)
            axes[i].set_title(f"Column {col}")
            axes[i].legend()

        plt.tight_layout()
        plt.show()
