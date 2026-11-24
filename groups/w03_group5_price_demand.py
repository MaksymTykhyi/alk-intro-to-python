"""Group 5 (w03) - price_demand: do cheaper products sell more? A log-log regression of units sold on price.

On a log-log scale the slope of  log(units) ~ log(price)  is the PRICE ELASTICITY:
-0.5 means "10% cheaper -> about 5% more units".

Authors: Max, Yehor, Natan, Jan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ml_base import Analysis


class PriceDemand(Analysis):
    """Price elasticity of demand across all products: log(units) explained by log(avg_price).

    Usage:
        model = PriceDemand(load_products()).fit()
        model.results_                  # {"elasticity": ..., "intercept": ..., "r2": ..., "n_products": ...}
        model.predict_units(2.50)       # units a typical product at £2.50 sells in a year
        model.search("heart")           # products whose description contains "heart"
        model.plot()                    # log-log scatter + the regression line
    """
    required_columns = ["avg_price", "units"]

    def fit(self):
        """Fit a LinearRegression of np.log(units) on np.log(avg_price).

        Only rows where both avg_price > 0 and units > 0 are used (log of 0 or a negative number is impossible).
        Stores the fitted scikit-learn model in self.model_ and
        self.results_ = {"elasticity", "intercept", "r2", "n_products"} (2 decimals).

        Returns:
            self (so you can write PriceDemand(products).fit())
        """
        usable = self.data[(self.data["avg_price"] > 0) & (self.data["units"] > 0)]
        if len(usable) < 2:
            raise ValueError("need at least 2 products with a positive price and positive units")

        X = pd.DataFrame({"log_price": np.log(usable["avg_price"])})     # a TABLE (2D), as scikit-learn wants
        y = np.log(usable["units"])

        self.model_ = LinearRegression().fit(X, y)
        self.results_ = {
            "elasticity": round(float(self.model_.coef_[0]), 2),
            "intercept": round(float(self.model_.intercept_), 2),
            "r2": round(float(self.model_.score(X, y)), 2),
            "n_products": int(len(usable)),
        }
        return self

    def predict_units(self, price):
        """Units a TYPICAL product at this price sells in a year: exp(intercept + elasticity * log(price)), rounded.

        Uses the unrounded coefficients of the model (results_ only shows them rounded).

        Args:
            price: price of one piece in GBP, a number > 0

        Returns:
            int - predicted units per year

        Raises:
            RuntimeError: if fit() was not called
            TypeError: if price is not a number
            ValueError: if price <= 0
        """
        self.check_fitted()
        if isinstance(price, bool) or not isinstance(price, (int, float, np.integer, np.floating)):
            raise TypeError(f"price must be a number, got {type(price).__name__}")
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")

        log_units = self.model_.intercept_ + self.model_.coef_[0] * np.log(price)
        return int(round(np.exp(log_units)))

    def search(self, word):
        """Products whose description contains `word` (not case sensitive), with the model's prediction next to reality.

        Args:
            word: a piece of text, e.g. "heart"

        Returns:
            DataFrame with the columns description, avg_price, units, predicted_units,
            sorted by units (best sellers first). Empty if nothing matches.

        Raises:
            RuntimeError: if fit() was not called
            TypeError: if word is not a string
            ValueError: if word is empty, or the data has no `description` column
        """
        self.check_fitted()
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {type(word).__name__}")
        if not word.strip():
            raise ValueError("word must not be empty")
        if "description" not in self.data.columns:
            raise ValueError("the data has no `description` column to search in")

        found = self.data[self.data["description"].str.contains(word.strip(), case=False, regex=False, na=False)]
        found = found[found["avg_price"] > 0]
        table = found[["description", "avg_price", "units"]].copy()
        table["predicted_units"] = [self.predict_units(float(p)) for p in table["avg_price"]]
        return table.sort_values("units", ascending=False)

    def plot(self):
        """Log-log scatter of all products (price vs units) + the red regression line.

        Returns:
            the matplotlib Axes (on a page: st.pyplot(model.plot().figure))

        Raises:
            RuntimeError: if fit() was not called
        """
        self.check_fitted()
        usable = self.data[(self.data["avg_price"] > 0) & (self.data["units"] > 0)]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(usable["avg_price"], usable["units"], s=8, alpha=0.3, label="products")

        prices = np.geomspace(usable["avg_price"].min(), usable["avg_price"].max(), 100)   # evenly spaced on a log axis
        line = np.exp(self.model_.intercept_ + self.model_.coef_[0] * np.log(prices))
        ax.plot(prices, line, color="red", linewidth=2,
                label=f"regression: elasticity {self.results_['elasticity']}, R² {self.results_['r2']}")

        ax.set(xscale="log", yscale="log", xlabel="average price [GBP, log scale]",
               ylabel="units sold in a year [log scale]", title="Do cheaper products sell more?")
        ax.legend()
        return ax
