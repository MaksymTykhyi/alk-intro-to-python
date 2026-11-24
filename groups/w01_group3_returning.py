"""Group 3 (w01), Card 3. Estimate which gift-shop customers will return.

Authors: Maciej, Maksym, Itan, Szymon.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ml_base import Analysis


class ReturnPredictor(Analysis):
    """Fit a return classifier on historical customer features.

    Each row represents a customer. Use features measured before the follow-up
    period and a binary `returned` column measured during that period.

    Example:
        model = ReturnPredictor(customers).fit()
        model.probability(orders=5, recency_days=30)
    """

    required_columns = ["returned"]

    def __init__(self, customers, features=("orders", "recency_days")):
        """Copy the table and select a nonempty sequence of feature names.

        Raises TypeError for a non-DataFrame input and ValueError for missing
        columns, invalid feature names, duplicates or use of the target.
        """
        super().__init__(customers)
        if isinstance(features, str):
            raise ValueError("features must be a sequence of column names, such as ('orders',)")
        try:
            self.features = tuple(features)
        except TypeError as exc:
            raise ValueError("features must be a sequence of column names") from exc
        if not self.features or not all(isinstance(name, str) for name in self.features):
            raise ValueError("provide at least one feature name and use strings for all names")
        if len(set(self.features)) != len(self.features):
            raise ValueError("feature names must be unique")
        if "returned" in self.features:
            raise ValueError("'returned' is the target and cannot be a feature")
        missing = [name for name in self.features if name not in self.data.columns]
        if missing:
            raise ValueError(f"data is missing feature columns: {missing}")

    def fit(self):
        """Train logistic regression and score it on a fixed 25% test split.

        Set results_ to the two test accuracies and the train and test sizes.
        The baseline always predicts the most common training outcome. Return
        self. Raise ValueError for invalid features or unusable training labels.
        """
        try:
            features = self.data.loc[:, list(self.features)].astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("feature columns must contain numeric values") from exc
        values = features.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("feature values must be finite and non-negative")

        target = self.data["returned"]
        if not target.isin([0, 1]).all():
            raise ValueError("'returned' must contain only 0 and 1")
        if len(target) < 3:
            raise ValueError("at least three customers are needed for a train/test split")
        x_train, x_test, y_train, y_test = train_test_split(
            features, target.astype(int), test_size=0.25, random_state=0
        )
        if y_train.nunique() != 2:
            raise ValueError("the training split needs both outcomes, 0 and 1")

        model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
        predictions = model.predict(x_test)
        majority_class = y_train.mode().iloc[0]
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

        self.model_ = model
        self._confusion = pd.DataFrame(
            matrix,
            index=["actual 0", "actual 1"],
            columns=["predicted 0", "predicted 1"],
        )
        self.results_ = {
            "accuracy": float((y_test == predictions).mean()),
            "baseline_accuracy": float((y_test == majority_class).mean()),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        return self

    def probability(self, **values):
        """Return the probability of return for one customer, to three decimals.

        Pass one finite, non-negative numeric value per fitted feature. Raise
        ValueError for invalid inputs or RuntimeError before fit().
        """
        self.check_fitted()
        missing = [name for name in self.features if name not in values]
        extra = [name for name in values if name not in self.features]
        if missing or extra:
            raise ValueError(f"expected values for {self.features}; missing: {missing}, unexpected: {extra}")
        try:
            inputs = np.array([values[name] for name in self.features], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("each feature must have one numeric value") from exc
        if inputs.ndim != 1 or not np.isfinite(inputs).all() or (inputs < 0).any():
            raise ValueError("each feature must have one finite, non-negative value")

        row = pd.DataFrame([inputs], columns=list(self.features))
        return round(float(self.model_.predict_proba(row)[0, 1]), 3)

    def confusion(self):
        """Return a copy of the 2×2 test confusion matrix (actual × predicted).

        Rows and columns always appear in 0, 1 order. Raise RuntimeError before
        fit().
        """
        self.check_fitted()
        return self._confusion.copy()
