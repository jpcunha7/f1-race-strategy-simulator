"""Visualization module for F1 Race Strategy Simulator.

Professional F1-themed visualizations with consistent styling:
- F1 red color scheme (#FF1E1E)
- Dark theme (plotly_dark)
- Clean, professional charts suitable for race engineers

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

# F1 Professional Color Palette
F1_RED = "#FF1E1E"
F1_BLUE = "#1E90FF"
F1_GREEN = "#00D856"
F1_YELLOW = "#FFA800"
F1_PURPLE = "#9B4DFF"
F1_COLORS = [F1_RED, F1_BLUE, F1_GREEN, F1_YELLOW, F1_PURPLE]


def plot_degradation_curves(
    models: dict[str, DegradationModel],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot degradation curves with confidence bands.

    Professional visualization showing tire degradation patterns with model type.
    """
    fig = go.Figure()

    stint_ages = np.arange(1, 41)
    color_idx = 0

    for compound, model in models.items():
        color = F1_COLORS[color_idx % len(F1_COLORS)]

        predictions = [model.predict(age) for age in stint_ages]
        upper_bound = [model.predict(age) + 2 * model.deg_rate_std * age for age in stint_ages]
        lower_bound = [model.predict(age) - 2 * model.deg_rate_std * age for age in stint_ages]

        # Main line
        fig.add_trace(
            go.Scatter(
                x=stint_ages,
                y=predictions,
                mode="lines",
                name=f"{compound} ({model.model_type})",
                line=dict(width=3, color=color),
                hovertemplate="<b>%{fullData.name}</b><br>" +
                             "Stint Age: %{x} laps<br>" +
                             "Lap Time: %{y:.2f}s<extra></extra>",
            )
        )

        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=list(stint_ages) + list(reversed(stint_ages)),
                y=upper_bound + list(reversed(lower_bound)),
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        color_idx += 1

    fig.update_layout(
        title=dict(
            text="Tire Degradation Analysis",
            font=dict(size=20, color="white"),
        ),
        xaxis=dict(
            title="Stint Age (laps)",
            gridcolor="rgba(255,255,255,0.1)",
            showgrid=True,
        ),
        yaxis=dict(
            title="Lap Time (seconds)",
            gridcolor="rgba(255,255,255,0.1)",
            showgrid=True,
        ),
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
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
    """Plot professional pit window analysis."""
    analysis_data = pit_analysis["analysis"]

    pit_laps = [a["pit_lap"] for a in analysis_data]
    mean_times = [a["mean_time"] for a in analysis_data]
    std_times = [a["std_time"] for a in analysis_data]

    fig = go.Figure()

    # Mean line with error bars
    fig.add_trace(
        go.Scatter(
            x=pit_laps,
            y=mean_times,
            mode="lines+markers",
            name="Expected Time",
            line=dict(width=3, color=F1_RED),
            marker=dict(size=8, color=F1_RED),
            error_y=dict(
                type="data",
                array=std_times,
                visible=True,
                color="rgba(255,30,30,0.3)",
            ),
            hovertemplate="Lap %{x}<br>Time: %{y:.2f}s<extra></extra>",
        )
    )

    # Find optimal pit lap
    optimal_idx = np.argmin(mean_times)
    optimal_lap = pit_laps[optimal_idx]
    optimal_time = mean_times[optimal_idx]

    # Mark optimal zone
    fig.add_vline(
        x=optimal_lap,
        line_dash="dot",
        line_color=F1_GREEN,
        line_width=2,
        annotation=dict(
            text=f"Optimal: Lap {optimal_lap} ({optimal_time:.1f}s)",
            font=dict(color=F1_GREEN, size=12),
        ),
    )

    fig.update_layout(
        title=dict(
            text=f"Pit Window Analysis: {pit_analysis['compound1']} → {pit_analysis['compound2']}",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Pit Stop Lap",
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            title="Expected Race Time (seconds)",
            gridcolor="rgba(255,255,255,0.1)",
        ),
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        hovermode="x",
    )

    return fig


def plot_strategy_comparison(
    comparison_df: pd.DataFrame,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot professional strategy comparison bar chart."""
    fig = go.Figure()

    # Sort by mean time
    df_sorted = comparison_df.sort_values("Mean Time (s)")

    # Color gradient: best=green, worst=red
    n_strategies = len(df_sorted)
    colors = []
    for i in range(n_strategies):
        if i == 0:
            colors.append(F1_GREEN)  # Best strategy
        elif i == 1:
            colors.append(F1_BLUE)
        else:
            colors.append(F1_RED)  # Others

    fig.add_trace(
        go.Bar(
            x=df_sorted["Strategy"],
            y=df_sorted["Mean Time (s)"],
            error_y=dict(
                type="data",
                array=df_sorted["Std Time (s)"],
                visible=True,
                color="rgba(255,255,255,0.3)",
            ),
            marker=dict(
                color=colors,
                line=dict(color="white", width=1),
            ),
            hovertemplate="<b>%{x}</b><br>" +
                         "Mean: %{y:.2f}s<br>" +
                         "<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Strategy Performance Comparison",
            font=dict(size=20, color="white"),
        ),
        xaxis=dict(
            title="Strategy",
            tickangle=-45,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            title="Mean Race Time (seconds)",
            gridcolor="rgba(255,255,255,0.1)",
        ),
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
    )

    return fig


def plot_risk_profiles(
    risk_profiles: dict,
    top_n: int = 5,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Plot risk profiles for top strategies.

    Shows best-case, mean, and worst-case scenarios for each strategy.
    """
    # Sort by mean time
    sorted_profiles = sorted(
        risk_profiles.items(),
        key=lambda x: x[1].mean_time
    )[:top_n]

    strategies = [name for name, _ in sorted_profiles]
    means = [profile.mean_time for _, profile in sorted_profiles]
    best_cases = [profile.best_case for _, profile in sorted_profiles]
    worst_cases = [profile.worst_case for _, profile in sorted_profiles]

    fig = go.Figure()

    # Error bars showing best/worst case
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=means,
            name="Mean Time",
            marker=dict(color=F1_BLUE),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[w - m for w, m in zip(worst_cases, means)],
                arrayminus=[m - b for m, b in zip(means, best_cases)],
                color="rgba(255,255,255,0.3)",
            ),
            hovertemplate="<b>%{x}</b><br>" +
                         "Mean: %{y:.2f}s<br>" +
                         "<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Strategy Risk Analysis (Best/Mean/Worst Case)",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Strategy",
            tickangle=-45,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            title="Race Time (seconds)",
            gridcolor="rgba(255,255,255,0.1)",
        ),
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
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
