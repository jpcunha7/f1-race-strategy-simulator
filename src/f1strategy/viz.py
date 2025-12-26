"""Visualization module for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1strategy.degrade_model import DegradationModel
from f1strategy.simulator import SimulationResult

logger = logging.getLogger(__name__)


def plot_degradation_curves(
    models: dict[str, DegradationModel],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot degradation curves with confidence bands."""
    fig = go.Figure()

    stint_ages = np.arange(1, 41)

    for compound, model in models.items():
        predictions = [model.predict(age) for age in stint_ages]
        upper_bound = [model.predict(age) + 2 * model.deg_rate_std * age for age in stint_ages]
        lower_bound = [model.predict(age) - 2 * model.deg_rate_std * age for age in stint_ages]

        # Main line
        fig.add_trace(
            go.Scatter(
                x=stint_ages,
                y=predictions,
                mode="lines",
                name=compound,
                line=dict(width=3),
            )
        )

        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=list(stint_ages) + list(reversed(stint_ages)),
                y=upper_bound + list(reversed(lower_bound)),
                fill="toself",
                fillcolor="rgba(0,100,200,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Tire Degradation Curves",
        xaxis_title="Stint Age (laps)",
        yaxis_title="Lap Time (seconds)",
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        hovermode="x unified",
    )

    return fig


def plot_race_time_distributions(
    results_dict: dict[str, list[SimulationResult]],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot distribution of total race times for each strategy."""
    fig = go.Figure()

    for strategy_name, results in results_dict.items():
        times = [r.total_time for r in results]

        fig.add_trace(
            go.Histogram(
                x=times,
                name=strategy_name,
                opacity=0.7,
                nbinsx=30,
            )
        )

    fig.update_layout(
        title="Race Time Distributions",
        xaxis_title="Total Race Time (seconds)",
        yaxis_title="Frequency",
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        barmode="overlay",
    )

    return fig


def plot_cumulative_distribution(
    results_dict: dict[str, list[SimulationResult]],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot cumulative distribution function."""
    fig = go.Figure()

    for strategy_name, results in results_dict.items():
        times = sorted([r.total_time for r in results])
        cdf = np.arange(1, len(times) + 1) / len(times)

        fig.add_trace(
            go.Scatter(
                x=times,
                y=cdf,
                mode="lines",
                name=strategy_name,
                line=dict(width=3),
            )
        )

    fig.update_layout(
        title="Cumulative Distribution of Race Times",
        xaxis_title="Total Race Time (seconds)",
        yaxis_title="Cumulative Probability",
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        hovermode="x unified",
    )

    return fig


def plot_pit_window_heatmap(
    pit_analysis: dict,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot heatmap of expected race time vs pit lap."""
    analysis_data = pit_analysis["analysis"]

    pit_laps = [a["pit_lap"] for a in analysis_data]
    mean_times = [a["mean_time"] for a in analysis_data]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pit_laps,
            y=mean_times,
            mode="lines+markers",
            name="Mean Race Time",
            line=dict(width=3, color="#FF1E1E"),
            marker=dict(size=8),
        )
    )

    # Find optimal pit lap
    optimal_idx = np.argmin(mean_times)
    optimal_lap = pit_laps[optimal_idx]

    fig.add_vline(
        x=optimal_lap,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Optimal: Lap {optimal_lap}",
    )

    fig.update_layout(
        title=f"Pit Window Analysis: {pit_analysis['compound1']} -> {pit_analysis['compound2']}",
        xaxis_title="Pit Lap",
        yaxis_title="Expected Race Time (seconds)",
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
    )

    return fig


def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot bar chart comparing strategies."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_df["Strategy"],
            y=comparison_df["Mean Time (s)"],
            error_y=dict(
                type="data",
                array=comparison_df["Std Time (s)"],
                visible=True,
            ),
            marker_color="#1E90FF",
        )
    )

    fig.update_layout(
        title="Strategy Comparison",
        xaxis_title="Strategy",
        yaxis_title="Mean Race Time (seconds)",
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        xaxis_tickangle=-45,
    )

    return fig


def plot_sensitivity_analysis(
    base_results: list[SimulationResult],
    varied_results: dict[str, list[SimulationResult]],
    parameter_name: str,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot sensitivity to a parameter."""
    fig = go.Figure()

    base_times = [r.total_time for r in base_results]

    fig.add_trace(
        go.Box(
            y=base_times,
            name="Baseline",
            marker_color="#1E90FF",
        )
    )

    for variant_name, results in varied_results.items():
        times = [r.total_time for r in results]

        fig.add_trace(
            go.Box(
                y=times,
                name=variant_name,
                marker_color="#FF1E1E",
            )
        )

    fig.update_layout(
        title=f"Sensitivity Analysis: {parameter_name}",
        xaxis_title="Scenario",
        yaxis_title="Total Race Time (seconds)",
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
    )

    return fig
