"""Tire degradation modeling for F1 Race Strategy Simulator.

Enhanced with automatic model selection (linear/quadratic/piecewise) using
cross-validation to choose the best fit for each compound's degradation pattern.

Physics reasoning: Tire degradation can follow different patterns:
- Linear: Constant rate of grip loss (most common)
- Quadratic: Accelerating degradation (cliff behavior)
- Piecewise: Different rates in early vs late stint

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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
            a = self.coefficients.get("a", 0)
            return self.baseline_laptime + self.deg_rate * stint_age + a * (stint_age**2)
        elif self.model_type == "piecewise":
            # Piecewise: different rates before/after breakpoint
            breakpoint = self.coefficients.get("breakpoint", 15)
            early_rate = self.coefficients.get("early_rate", self.deg_rate)
            late_rate = self.coefficients.get("late_rate", self.deg_rate)

            if stint_age <= breakpoint:
                return self.baseline_laptime + early_rate * stint_age
            else:
                # Continue from breakpoint with different rate
                early_contrib = early_rate * breakpoint
                late_contrib = late_rate * (stint_age - breakpoint)
                return self.baseline_laptime + early_contrib + late_contrib
        else:
            return self.baseline_laptime + self.deg_rate * stint_age

    def predict_with_noise(self, stint_age: int, rng: np.random.Generator) -> float:
        """Predict lap time with uncertainty."""
        base_prediction = self.predict(stint_age)
        noise = rng.normal(0, self.deg_rate_std * stint_age)
        return base_prediction + noise


def remove_outliers(data: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers using z-score method."""
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]


def fit_linear_model(X: np.ndarray, y: np.ndarray) -> tuple[float, float, dict, float]:
    """Fit linear degradation model.

    Returns: (baseline, deg_rate, coefficients, r_squared)
    """
    model = HuberRegressor()
    model.fit(X, y)

    baseline = float(model.intercept_)
    deg_rate = float(model.coef_[0])

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    coefficients = {"intercept": baseline, "slope": deg_rate}

    return baseline, deg_rate, coefficients, r_squared


def fit_quadratic_model(X: np.ndarray, y: np.ndarray) -> tuple[float, float, dict, float]:
    """Fit quadratic degradation model.

    Returns: (baseline, deg_rate, coefficients, r_squared)
    """
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

    coefficients = {"intercept": baseline, "slope": deg_rate, "a": quad_coef}

    return baseline, deg_rate, coefficients, r_squared


def fit_piecewise_model(
    X: np.ndarray, y: np.ndarray, breakpoint: int = 15
) -> tuple[float, float, dict, float]:
    """Fit piecewise linear degradation model.

    Two different degradation rates: early stint vs late stint.

    Returns: (baseline, avg_deg_rate, coefficients, r_squared)
    """
    X_flat = X.flatten()

    # Split data at breakpoint
    early_mask = X_flat <= breakpoint
    late_mask = X_flat > breakpoint

    if np.sum(early_mask) < 3 or np.sum(late_mask) < 3:
        # Not enough data for piecewise, fall back to linear
        return fit_linear_model(X, y)

    # Fit early stint
    X_early = X[early_mask].reshape(-1, 1)
    y_early = y[early_mask]
    early_model = LinearRegression()
    early_model.fit(X_early, y_early)

    baseline = float(early_model.intercept_)
    early_rate = float(early_model.coef_[0])

    # Fit late stint (constrained to match at breakpoint)
    X_late = X[late_mask].reshape(-1, 1)
    y_late = y[late_mask]

    # Expected value at breakpoint from early model
    breakpoint_value = baseline + early_rate * breakpoint

    # Fit late segment
    late_model = LinearRegression()
    late_model.fit(X_late, y_late)
    late_rate = float(late_model.coef_[0])

    # Calculate predictions for all data
    y_pred = np.zeros_like(y)
    y_pred[early_mask] = baseline + early_rate * X_flat[early_mask]

    late_contrib = early_rate * breakpoint + late_rate * (X_flat[late_mask] - breakpoint)
    y_pred[late_mask] = baseline + late_contrib

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Average degradation rate for compatibility
    avg_deg_rate = (early_rate + late_rate) / 2

    coefficients = {
        "intercept": baseline,
        "breakpoint": breakpoint,
        "early_rate": early_rate,
        "late_rate": late_rate,
    }

    return baseline, avg_deg_rate, coefficients, r_squared


def select_best_degradation_model(
    X: np.ndarray,
    y: np.ndarray,
    config: StrategyConfig,
) -> tuple[str, float, float, dict, float]:
    """Select best degradation model using cross-validation.

    Strategy reasoning: Different tires/tracks exhibit different degradation
    patterns. Automatic model selection ensures we use the most appropriate
    model for each compound.

    Args:
        X: Stint ages
        y: Lap times
        config: Strategy configuration

    Returns:
        (best_model_type, baseline, deg_rate, coefficients, r_squared)
    """
    if len(X) < config.cv_folds * 2:
        # Not enough data for CV, default to linear
        logger.warning("Insufficient data for CV, using linear model")
        baseline, deg_rate, coeffs, r2 = fit_linear_model(X, y)
        return "linear", baseline, deg_rate, coeffs, r2

    # Try all three models with cross-validation
    kfold = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)

    model_scores = {}

    # Linear model
    cv_scores_linear = []
    for train_idx, val_idx in kfold.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        baseline, deg_rate, _, _ = fit_linear_model(X_train, y_train)
        y_pred = baseline + deg_rate * X_val.flatten()
        mse = mean_squared_error(y_val, y_pred)
        cv_scores_linear.append(mse)

    model_scores["linear"] = np.mean(cv_scores_linear)

    # Quadratic model
    cv_scores_quad = []
    for train_idx, val_idx in kfold.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        baseline, deg_rate, coeffs, _ = fit_quadratic_model(X_train, y_train)
        X_val_flat = X_val.flatten()
        y_pred = baseline + deg_rate * X_val_flat + coeffs["a"] * (X_val_flat**2)
        mse = mean_squared_error(y_val, y_pred)
        cv_scores_quad.append(mse)

    model_scores["quadratic"] = np.mean(cv_scores_quad)

    # Piecewise model
    cv_scores_piece = []
    for train_idx, val_idx in kfold.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        baseline, _, coeffs, _ = fit_piecewise_model(X_train, y_train, breakpoint=15)

        # Predict validation set
        X_val_flat = X_val.flatten()
        y_pred = np.zeros_like(y_val)

        breakpoint = coeffs.get("breakpoint", 15)
        early_rate = coeffs.get("early_rate", 0)
        late_rate = coeffs.get("late_rate", 0)

        early_mask = X_val_flat <= breakpoint
        late_mask = X_val_flat > breakpoint

        if np.any(early_mask):
            y_pred[early_mask] = baseline + early_rate * X_val_flat[early_mask]

        if np.any(late_mask):
            early_contrib = early_rate * breakpoint
            late_contrib = late_rate * (X_val_flat[late_mask] - breakpoint)
            y_pred[late_mask] = baseline + early_contrib + late_contrib

        mse = mean_squared_error(y_val, y_pred)
        cv_scores_piece.append(mse)

    model_scores["piecewise"] = np.mean(cv_scores_piece)

    # Select best model
    best_model_type = min(model_scores, key=model_scores.get)

    logger.info(
        f"Model selection CV scores: "
        f"linear={model_scores['linear']:.3f}, "
        f"quadratic={model_scores['quadratic']:.3f}, "
        f"piecewise={model_scores['piecewise']:.3f} "
        f"-> selected: {best_model_type}"
    )

    # Refit best model on full data
    if best_model_type == "linear":
        baseline, deg_rate, coeffs, r2 = fit_linear_model(X, y)
    elif best_model_type == "quadratic":
        baseline, deg_rate, coeffs, r2 = fit_quadratic_model(X, y)
    else:  # piecewise
        baseline, deg_rate, coeffs, r2 = fit_piecewise_model(X, y)

    return best_model_type, baseline, deg_rate, coeffs, r2


def fit_degradation_model(
    stint_data: pd.DataFrame,
    compound: str,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> Optional[DegradationModel]:
    """Fit degradation model for a specific compound."""
    try:
        # Filter for this compound
        compound_data = stint_data[stint_data["Compound"] == compound].copy()

        if compound_data.empty:
            logger.warning(f"No data for compound {compound}")
            return None

        # Remove outliers and pit laps
        clean_data = compound_data[~compound_data["IsOutlier"]].copy()

        if len(clean_data) < 5:
            logger.warning(f"Insufficient clean data for {compound}")
            return None

        # Remove statistical outliers based on lap time
        clean_data = remove_outliers(clean_data, "LapTime", config.outlier_threshold)

        if len(clean_data) < 5:
            logger.warning(f"Insufficient data after outlier removal for {compound}")
            return None

        X = clean_data["StintAge"].values.reshape(-1, 1)
        y = clean_data["LapTime"].values

        # Auto-select best model or use specified type
        if config.auto_select_degradation_model:
            model_type, baseline, deg_rate, coefficients, r_squared = select_best_degradation_model(
                X, y, config
            )
        else:
            # Use specified model type
            if config.degradation_model_type == "linear":
                baseline, deg_rate, coefficients, r_squared = fit_linear_model(X, y)
                model_type = "linear"
            elif config.degradation_model_type == "quadratic":
                baseline, deg_rate, coefficients, r_squared = fit_quadratic_model(X, y)
                model_type = "quadratic"
            else:  # piecewise
                baseline, deg_rate, coefficients, r_squared = fit_piecewise_model(X, y)
                model_type = "piecewise"

        # Estimate uncertainty from residuals
        # Recreate predictions based on model type
        if model_type == "linear":
            y_pred = baseline + deg_rate * X.flatten()
        elif model_type == "quadratic":
            y_pred = baseline + deg_rate * X.flatten() + coefficients["a"] * (X.flatten() ** 2)
        else:  # piecewise
            X_flat = X.flatten()
            breakpoint = coefficients.get("breakpoint", 15)
            early_rate = coefficients.get("early_rate", deg_rate)
            late_rate = coefficients.get("late_rate", deg_rate)

            y_pred = np.zeros_like(X_flat, dtype=float)
            early_mask = X_flat <= breakpoint
            late_mask = X_flat > breakpoint

            if np.any(early_mask):
                y_pred[early_mask] = baseline + early_rate * X_flat[early_mask]
            if np.any(late_mask):
                early_contrib = early_rate * breakpoint
                late_contrib = late_rate * (X_flat[late_mask] - breakpoint)
                y_pred[late_mask] = baseline + early_contrib + late_contrib

        residuals = y - y_pred
        deg_rate_std = float(np.std(residuals / X.flatten()))

        logger.info(
            f"{compound}: model={model_type}, baseline={baseline:.2f}s, "
            f"deg_rate={deg_rate:.3f}s/lap, R²={r_squared:.3f}"
        )

        return DegradationModel(
            compound=compound,
            model_type=model_type,
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
    compounds = stint_data["Compound"].unique()
    compounds = [c for c in compounds if c != "UNKNOWN"]

    models = {}
    for compound in compounds:
        model = fit_degradation_model(stint_data, compound, config)
        if model is not None:
            models[compound] = model

    logger.info(f"Fitted models for {len(models)} compounds: {list(models.keys())}")
    return models
