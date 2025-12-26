"""Tire degradation modeling for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, HuberRegressor

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class DegradationModel:
    """Fitted degradation model for a tire compound."""

    compound: str
    model_type: str
    baseline_laptime: float  # seconds
    deg_rate: float  # seconds per lap
    deg_rate_std: float  # uncertainty in degradation rate
    r_squared: float
    n_samples: int
    coefficients: dict  # model-specific coefficients

    def predict(self, stint_age: int) -> float:
        """Predict lap time at given stint age."""
        if self.model_type == "linear":
            return self.baseline_laptime + self.deg_rate * stint_age
        elif self.model_type == "quadratic":
            a = self.coefficients.get('a', 0)
            return self.baseline_laptime + self.deg_rate * stint_age + a * (stint_age ** 2)
        else:
            return self.baseline_laptime + self.deg_rate * stint_age

    def predict_with_noise(self, stint_age: int, rng: np.random.Generator) -> float:
        """Predict lap time with uncertainty."""
        base_prediction = self.predict(stint_age)
        noise = rng.normal(0, self.deg_rate_std * stint_age)
        return base_prediction + noise


def remove_outliers(
    data: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> pd.DataFrame:
    """Remove outliers using z-score method."""
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]


def fit_degradation_model(
    stint_data: pd.DataFrame,
    compound: str,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> Optional[DegradationModel]:
    """Fit degradation model for a specific compound."""
    try:
        # Filter for this compound
        compound_data = stint_data[stint_data['Compound'] == compound].copy()

        if compound_data.empty:
            logger.warning(f"No data for compound {compound}")
            return None

        # Remove outliers and pit laps
        clean_data = compound_data[~compound_data['IsOutlier']].copy()

        if len(clean_data) < 5:
            logger.warning(f"Insufficient clean data for {compound}")
            return None

        # Remove statistical outliers based on lap time
        clean_data = remove_outliers(clean_data, 'LapTime', config.outlier_threshold)

        if len(clean_data) < 5:
            logger.warning(f"Insufficient data after outlier removal for {compound}")
            return None

        X = clean_data['StintAge'].values.reshape(-1, 1)
        y = clean_data['LapTime'].values

        # Fit model based on type
        if config.degradation_model_type == "linear":
            # Use Huber regression for robustness
            model = HuberRegressor()
            model.fit(X, y)

            baseline = float(model.intercept_)
            deg_rate = float(model.coef_[0])

            # Calculate R-squared
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Estimate uncertainty
            residuals = y - y_pred
            deg_rate_std = float(np.std(residuals / X.flatten()))

            coefficients = {'intercept': baseline, 'slope': deg_rate}

        elif config.degradation_model_type == "quadratic":
            X_quad = np.column_stack([X, X**2])
            model = LinearRegression()
            model.fit(X_quad, y)

            baseline = float(model.intercept_)
            deg_rate = float(model.coef_[0])
            quad_coef = float(model.coef_[1])

            y_pred = model.predict(X_quad)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            residuals = y - y_pred
            deg_rate_std = float(np.std(residuals / X.flatten()))

            coefficients = {'intercept': baseline, 'slope': deg_rate, 'a': quad_coef}

        else:  # piecewise - simplified as linear for now
            model = LinearRegression()
            model.fit(X, y)

            baseline = float(model.intercept_)
            deg_rate = float(model.coef_[0])

            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            residuals = y - y_pred
            deg_rate_std = float(np.std(residuals / X.flatten()))

            coefficients = {'intercept': baseline, 'slope': deg_rate}

        logger.info(
            f"{compound}: baseline={baseline:.2f}s, "
            f"deg_rate={deg_rate:.3f}s/lap, R²={r_squared:.3f}"
        )

        return DegradationModel(
            compound=compound,
            model_type=config.degradation_model_type,
            baseline_laptime=baseline,
            deg_rate=deg_rate,
            deg_rate_std=max(deg_rate_std, 0.001),  # Ensure non-zero
            r_squared=r_squared,
            n_samples=len(clean_data),
            coefficients=coefficients,
        )

    except Exception as e:
        logger.error(f"Failed to fit model for {compound}: {e}")
        return None


def fit_all_compounds(
    stint_data: pd.DataFrame,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> dict[str, DegradationModel]:
    """Fit degradation models for all compounds in the data."""
    compounds = stint_data['Compound'].unique()
    compounds = [c for c in compounds if c != 'UNKNOWN']

    models = {}
    for compound in compounds:
        model = fit_degradation_model(stint_data, compound, config)
        if model is not None:
            models[compound] = model

    logger.info(f"Fitted models for {len(models)} compounds: {list(models.keys())}")
    return models
