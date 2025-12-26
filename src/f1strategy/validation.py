"""Model validation against historical race data for F1 Race Strategy Simulator.

This module provides critical credibility for strategy predictions by validating
degradation models against actual race performance. Uses actual stint data to
assess prediction accuracy and identify systematic errors.

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1strategy.degrade_model import fit_degradation_model

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Validation metrics for a degradation model."""

    compound: str
    mae: float  # mean absolute error (seconds)
    rmse: float  # root mean squared error (seconds)
    mae_by_stint_age: dict[int, float]  # MAE by stint age
    r_squared: float
    n_predictions: int
    traffic_laps_detected: int
    outlier_laps_detected: int


@dataclass
class ValidationResult:
    """Complete validation results for a race."""

    driver: str
    year: int
    event: str
    metrics_by_compound: dict[str, ValidationMetrics]
    overall_mae: float
    overall_rmse: float
    predictions_df: pd.DataFrame  # actual vs predicted for plotting


def identify_traffic_laps(lap_times: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    """Identify laps likely affected by traffic using statistical outlier detection.

    Physics reasoning: Traffic causes significant lap time increases beyond
    normal degradation variance. Uses z-score to detect anomalous laps.

    Args:
        lap_times: Array of lap times in seconds
        threshold: Z-score threshold for outlier detection

    Returns:
        Boolean array indicating traffic-affected laps
    """
    if len(lap_times) < 3:
        return np.zeros(len(lap_times), dtype=bool)

    z_scores = np.abs(stats.zscore(lap_times, nan_policy="omit"))
    return z_scores > threshold


def validate_race(
    year: int,
    event: str,
    driver: str,
    stint_data: pd.DataFrame,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> ValidationResult:
    """Validate degradation model against actual race performance.

    Strategy reasoning: This validation is critical for race engineers to trust
    model predictions. By fitting on training data and testing on held-out stints,
    we assess real-world accuracy and identify systematic biases.

    Process:
    1. Split stint data into train/test by stint
    2. Fit degradation model on training stints
    3. Predict lap times for test stints
    4. Calculate prediction errors (MAE, RMSE)
    5. Identify traffic/outlier laps that explain large errors

    Args:
        year: Race year
        event: Event name
        driver: Driver code
        stint_data: DataFrame with stint information
        config: Strategy configuration

    Returns:
        ValidationResult with comprehensive validation metrics
    """
    logger.info(f"Validating models for {driver} at {year} {event}")

    if len(stint_data) < config.validation_min_laps:
        raise ValueError(
            f"Insufficient data for validation: {len(stint_data)} laps "
            f"(minimum {config.validation_min_laps})"
        )

    # Get unique stints
    stints = stint_data["Stint"].unique()
    n_test_stints = max(1, int(len(stints) * config.validation_test_size))

    # Use last stints as test set (more realistic - predict later stints)
    train_stints = stints[:-n_test_stints]
    test_stints = stints[-n_test_stints:]

    train_data = stint_data[stint_data["Stint"].isin(train_stints)]
    test_data = stint_data[stint_data["Stint"].isin(test_stints)]

    logger.info(
        f"Train/test split: {len(train_stints)} train stints, " f"{len(test_stints)} test stints"
    )

    # Fit models on training data
    compounds = train_data["Compound"].unique()
    compounds = [c for c in compounds if c != "UNKNOWN"]

    metrics_by_compound = {}
    all_predictions = []

    for compound in compounds:
        # Fit model on training data
        model = fit_degradation_model(train_data, compound, config)

        if model is None:
            logger.warning(f"Could not fit model for {compound}, skipping")
            continue

        # Get test data for this compound
        compound_test = test_data[test_data["Compound"] == compound].copy()

        if len(compound_test) == 0:
            logger.warning(f"No test data for {compound}, skipping validation")
            continue

        # Remove already-flagged outliers
        clean_test = compound_test[~compound_test["IsOutlier"]].copy()

        if len(clean_test) < 3:
            logger.warning(f"Insufficient clean test data for {compound}")
            continue

        # Predict lap times
        predictions = []
        actuals = []
        stint_ages = []

        for _, row in clean_test.iterrows():
            predicted = model.predict(int(row["StintAge"]))
            actual = row["LapTime"]

            predictions.append(predicted)
            actuals.append(actual)
            stint_ages.append(int(row["StintAge"]))

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Identify traffic laps in test data
        residuals = actuals - predictions
        traffic_mask = identify_traffic_laps(residuals, config.traffic_detection_threshold)
        n_traffic = np.sum(traffic_mask)

        # Calculate metrics (excluding traffic laps)
        clean_mask = ~traffic_mask
        clean_predictions = predictions[clean_mask]
        clean_actuals = actuals[clean_mask]

        if len(clean_actuals) == 0:
            logger.warning(f"No clean predictions for {compound} after traffic removal")
            continue

        mae = mean_absolute_error(clean_actuals, clean_predictions)
        rmse = np.sqrt(mean_squared_error(clean_actuals, clean_predictions))

        # MAE by stint age
        mae_by_age = {}
        for age in np.unique(stint_ages):
            age_mask = (np.array(stint_ages) == age) & clean_mask
            if np.sum(age_mask) > 0:
                age_mae = mean_absolute_error(actuals[age_mask], predictions[age_mask])
                mae_by_age[int(age)] = float(age_mae)

        # R-squared on test set
        ss_res = np.sum((clean_actuals - clean_predictions) ** 2)
        ss_tot = np.sum((clean_actuals - np.mean(clean_actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = ValidationMetrics(
            compound=compound,
            mae=float(mae),
            rmse=float(rmse),
            mae_by_stint_age=mae_by_age,
            r_squared=float(r_squared),
            n_predictions=len(clean_actuals),
            traffic_laps_detected=int(n_traffic),
            outlier_laps_detected=0,  # Already filtered in IsOutlier
        )

        metrics_by_compound[compound] = metrics

        # Store predictions for plotting
        for i in range(len(predictions)):
            all_predictions.append(
                {
                    "Compound": compound,
                    "StintAge": stint_ages[i],
                    "Actual": actuals[i],
                    "Predicted": predictions[i],
                    "Residual": actuals[i] - predictions[i],
                    "IsTraffic": traffic_mask[i],
                }
            )

        logger.info(
            f"{compound}: MAE={mae:.3f}s, RMSE={rmse:.3f}s, "
            f"R²={r_squared:.3f}, traffic_laps={n_traffic}"
        )

    if not metrics_by_compound:
        raise ValueError("No valid validation metrics could be computed")

    # Overall metrics
    predictions_df = pd.DataFrame(all_predictions)
    clean_preds = predictions_df[~predictions_df["IsTraffic"]]

    overall_mae = mean_absolute_error(clean_preds["Actual"], clean_preds["Predicted"])
    overall_rmse = np.sqrt(mean_squared_error(clean_preds["Actual"], clean_preds["Predicted"]))

    logger.info(f"Overall validation: MAE={overall_mae:.3f}s, RMSE={overall_rmse:.3f}s")

    return ValidationResult(
        driver=driver,
        year=year,
        event=event,
        metrics_by_compound=metrics_by_compound,
        overall_mae=float(overall_mae),
        overall_rmse=float(overall_rmse),
        predictions_df=predictions_df,
    )


def create_validation_plots(
    validation_result: ValidationResult,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Create comprehensive validation visualization.

    Plots:
    1. Actual vs Predicted scatter (by compound)
    2. Residuals vs Stint Age
    3. Residual distribution

    Args:
        validation_result: Validation results
        config: Strategy configuration

    Returns:
        Plotly figure with validation plots
    """
    df = validation_result.predictions_df

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Actual vs Predicted Lap Times",
            "Residuals vs Stint Age",
            "Residual Distribution",
            "MAE by Compound",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "bar"}],
        ],
    )

    # 1. Actual vs Predicted scatter
    for compound in df["Compound"].unique():
        compound_df = df[df["Compound"] == compound]
        clean_df = compound_df[~compound_df["IsTraffic"]]

        fig.add_trace(
            go.Scatter(
                x=clean_df["Predicted"],
                y=clean_df["Actual"],
                mode="markers",
                name=compound,
                marker=dict(size=8, opacity=0.6),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Add perfect prediction line
    all_times = np.concatenate([df["Actual"], df["Predicted"]])
    min_time, max_time = all_times.min(), all_times.max()

    fig.add_trace(
        go.Scatter(
            x=[min_time, max_time],
            y=[min_time, max_time],
            mode="lines",
            line=dict(color="white", dash="dash", width=2),
            name="Perfect Prediction",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Residuals vs Stint Age
    for compound in df["Compound"].unique():
        compound_df = df[df["Compound"] == compound]
        clean_df = compound_df[~compound_df["IsTraffic"]]

        fig.add_trace(
            go.Scatter(
                x=clean_df["StintAge"],
                y=clean_df["Residual"],
                mode="markers",
                name=compound,
                marker=dict(size=8, opacity=0.6),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)

    # 3. Residual distribution
    clean_residuals = df[~df["IsTraffic"]]["Residual"]

    fig.add_trace(
        go.Histogram(
            x=clean_residuals,
            nbinsx=30,
            name="Residuals",
            marker_color="#FF1E1E",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. MAE by compound
    compounds = list(validation_result.metrics_by_compound.keys())
    maes = [validation_result.metrics_by_compound[c].mae for c in compounds]

    fig.add_trace(
        go.Bar(
            x=compounds,
            y=maes,
            marker_color="#1E90FF",
            name="MAE",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="Predicted Lap Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Actual Lap Time (s)", row=1, col=1)

    fig.update_xaxes(title_text="Stint Age (laps)", row=1, col=2)
    fig.update_yaxes(title_text="Residual (s)", row=1, col=2)

    fig.update_xaxes(title_text="Residual (s)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_xaxes(title_text="Compound", row=2, col=2)
    fig.update_yaxes(title_text="MAE (s)", row=2, col=2)

    fig.update_layout(
        title=f"Model Validation: {validation_result.driver} - "
        f"{validation_result.year} {validation_result.event}",
        template=config.plot_theme,
        height=800,
        width=config.plot_width,
        showlegend=True,
    )

    return fig


def generate_validation_report(
    validation_result: ValidationResult,
) -> pd.DataFrame:
    """Generate validation summary report.

    Args:
        validation_result: Validation results

    Returns:
        DataFrame with validation metrics by compound
    """
    report_data = []

    for compound, metrics in validation_result.metrics_by_compound.items():
        report_data.append(
            {
                "Compound": compound,
                "MAE (s)": f"{metrics.mae:.3f}",
                "RMSE (s)": f"{metrics.rmse:.3f}",
                "R²": f"{metrics.r_squared:.3f}",
                "Predictions": metrics.n_predictions,
                "Traffic Laps": metrics.traffic_laps_detected,
            }
        )

    # Add overall row
    report_data.append(
        {
            "Compound": "OVERALL",
            "MAE (s)": f"{validation_result.overall_mae:.3f}",
            "RMSE (s)": f"{validation_result.overall_rmse:.3f}",
            "R²": "-",
            "Predictions": len(validation_result.predictions_df),
            "Traffic Laps": validation_result.predictions_df["IsTraffic"].sum(),
        }
    )

    return pd.DataFrame(report_data)
