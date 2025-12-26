"""Undercut and overcut analysis for F1 Race Strategy Simulator.

This module implements professional undercut/overcut gain estimation, critical
for race strategists to identify optimal pit windows and reactive strategies.

Physics reasoning:
- Undercut gain comes from: (1) opponent's tire degradation, (2) your fresh tire pace
- Offset by: (1) pit loss time, (2) tire warmup penalty on fresh tires
- Overcut gain comes from: staying out on degrading tires while opponent suffers warmup

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1strategy.degrade_model import DegradationModel

logger = logging.getLogger(__name__)


@dataclass
class UndercutAnalysis:
    """Analysis of undercut opportunity at a specific lap."""

    pit_lap: int
    your_compound: str
    opponent_compound: str
    estimated_gain: float  # seconds gained (positive = advantage)
    confidence_interval: tuple[float, float]  # 95% CI
    breakdown: dict[str, float]  # detailed gain/loss components


@dataclass
class PitWindowRecommendation:
    """Pit window recommendation with optimal timing."""

    optimal_lap: int
    expected_gain: float
    window_start: int  # earliest lap with positive gain
    window_end: int  # latest lap with positive gain
    risk_assessment: str  # "low", "medium", "high"


def calculate_warmup_penalty(
    laps_into_stint: int,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> float:
    """Calculate tire warmup penalty for a given lap in stint.

    Physics reasoning: Fresh tires require 1-3 laps to reach optimal temperature.
    During warmup, grip is reduced, causing lap time penalties.

    Two models:
    1. Exponential decay: penalty = initial * exp(-lap / tau)
       - More realistic, gradual warmup
    2. Step function: fixed penalty for first N laps
       - Simpler, conservative estimate

    Args:
        laps_into_stint: Lap number within stint (1 = first lap on new tires)
        config: Strategy configuration

    Returns:
        Warmup penalty in seconds
    """
    if laps_into_stint < 1:
        return 0.0

    if config.warmup_model == "exponential":
        # Exponential decay model
        penalty = config.warmup_penalty_initial * np.exp(-laps_into_stint / config.warmup_decay_tau)
        return float(penalty)
    else:
        # Step function model
        if laps_into_stint <= config.warmup_step_laps:
            return config.warmup_penalty_initial
        else:
            return 0.0


def estimate_undercut_gain(
    current_lap: int,
    pit_lap: int,
    your_current_compound: str,
    your_new_compound: str,
    opponent_compound: str,
    your_stint_age: int,
    opponent_stint_age: int,
    degradation_models: dict[str, DegradationModel],
    config: StrategyConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
) -> UndercutAnalysis:
    """Estimate undercut gain if you pit at a given lap.

    Strategy reasoning: Undercut works when the time gained from fresh tires
    exceeds the pit loss. Key factors:
    1. Your tire degradation before pit
    2. Opponent's continued degradation while you're in pits
    3. Your warmup penalty on fresh tires
    4. Net pace difference after both on different compounds/ages

    Calculation:
    - Lap you pit: lose pit_loss seconds
    - Next lap: you have warmup penalty, opponent has +1 tire age degradation
    - Following laps: gap changes based on compound pace + degradation difference

    Args:
        current_lap: Current race lap
        pit_lap: Lap when you would pit
        your_current_compound: Your current tire compound
        your_new_compound: Compound you'll switch to
        opponent_compound: Opponent's current compound
        your_stint_age: Your current stint age
        opponent_stint_age: Opponent's current stint age
        degradation_models: Dictionary of degradation models
        config: Strategy configuration
        rng: Random number generator for uncertainty

    Returns:
        UndercutAnalysis with estimated gain and breakdown
    """
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    if pit_lap < current_lap:
        raise ValueError("Cannot pit in the past")

    # Get models
    your_current_model = degradation_models.get(your_current_compound)
    your_new_model = degradation_models.get(your_new_compound)
    opponent_model = degradation_models.get(opponent_compound)

    if not all([your_current_model, your_new_model, opponent_model]):
        raise ValueError("Missing degradation models for compounds")

    # Simulate next 5 laps after pit to estimate net gain
    n_laps_to_simulate = 5
    net_gain = 0.0
    breakdown = {}

    # 1. Pit loss
    pit_loss = rng.normal(config.pit_loss_mean, config.pit_loss_std)
    net_gain -= pit_loss
    breakdown["pit_loss"] = -pit_loss

    # 2. Simulate laps after pit
    your_age_after_pit = 0
    opponent_age_at_pit = opponent_stint_age + (pit_lap - current_lap)

    total_warmup_penalty = 0.0
    total_degradation_advantage = 0.0

    for lap_offset in range(1, n_laps_to_simulate + 1):
        your_age_after_pit += 1
        opponent_age_at_pit += 1

        # Your lap time (new compound, includes warmup)
        your_base_time = your_new_model.predict(your_age_after_pit)
        warmup = calculate_warmup_penalty(your_age_after_pit, config)
        your_time = your_base_time + warmup
        total_warmup_penalty += warmup

        # Opponent lap time (old compound, degrading)
        opponent_time = opponent_model.predict(opponent_age_at_pit)

        # Net gain this lap (positive = you're faster)
        lap_gain = opponent_time - your_time
        net_gain += lap_gain
        total_degradation_advantage += lap_gain

    breakdown["warmup_penalty"] = -total_warmup_penalty
    breakdown["degradation_advantage"] = total_degradation_advantage

    # Uncertainty: use model uncertainty
    uncertainty = (
        np.sqrt(your_new_model.deg_rate_std**2 + opponent_model.deg_rate_std**2)
        * n_laps_to_simulate
    )

    ci_lower = net_gain - 1.96 * uncertainty
    ci_upper = net_gain + 1.96 * uncertainty

    logger.info(
        f"Undercut analysis lap {pit_lap}: gain={net_gain:.2f}s "
        f"(pit_loss={pit_loss:.1f}s, warmup={total_warmup_penalty:.2f}s, "
        f"deg_adv={total_degradation_advantage:.2f}s)"
    )

    return UndercutAnalysis(
        pit_lap=pit_lap,
        your_compound=your_new_compound,
        opponent_compound=opponent_compound,
        estimated_gain=float(net_gain),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        breakdown=breakdown,
    )


def estimate_overcut_gain(
    current_lap: int,
    opponent_pit_lap: int,
    your_compound: str,
    opponent_new_compound: str,
    your_stint_age: int,
    opponent_stint_age_at_pit: int,
    degradation_models: dict[str, DegradationModel],
    config: StrategyConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
) -> UndercutAnalysis:
    """Estimate overcut gain if opponent pits and you stay out.

    Strategy reasoning: Overcut works when staying out on degrading tires
    still gives better lap times than opponent's warmup penalty and pit loss.

    Calculation inverse of undercut:
    - Opponent loses pit_loss seconds
    - Opponent has warmup penalty
    - You continue on degrading tires but no pit loss

    Args:
        current_lap: Current race lap
        opponent_pit_lap: Lap when opponent pits
        your_compound: Your current compound
        opponent_new_compound: Compound opponent switches to
        your_stint_age: Your current stint age
        opponent_stint_age_at_pit: Opponent's stint age when they pit
        degradation_models: Dictionary of degradation models
        config: Strategy configuration
        rng: Random number generator

    Returns:
        UndercutAnalysis with estimated overcut gain
    """
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    your_model = degradation_models.get(your_compound)
    opponent_new_model = degradation_models.get(opponent_new_compound)

    if not all([your_model, opponent_new_model]):
        raise ValueError("Missing degradation models for compounds")

    n_laps_to_simulate = 5
    net_gain = 0.0
    breakdown = {}

    # Opponent pays pit loss
    pit_loss = rng.normal(config.pit_loss_mean, config.pit_loss_std)
    net_gain += pit_loss  # You gain because they lose time
    breakdown["opponent_pit_loss"] = pit_loss

    # Simulate laps
    your_age = your_stint_age + (opponent_pit_lap - current_lap)
    opponent_age_after_pit = 0

    total_opponent_warmup = 0.0
    total_degradation_difference = 0.0

    for lap_offset in range(1, n_laps_to_simulate + 1):
        your_age += 1
        opponent_age_after_pit += 1

        # Your lap time (degrading)
        your_time = your_model.predict(your_age)

        # Opponent lap time (new tires, warmup penalty)
        opponent_base = opponent_new_model.predict(opponent_age_after_pit)
        warmup = calculate_warmup_penalty(opponent_age_after_pit, config)
        opponent_time = opponent_base + warmup
        total_opponent_warmup += warmup

        # Net gain (positive = you're faster)
        lap_gain = opponent_time - your_time
        net_gain += lap_gain
        total_degradation_difference += lap_gain

    breakdown["opponent_warmup_penalty"] = total_opponent_warmup
    breakdown["degradation_difference"] = total_degradation_difference

    uncertainty = (
        np.sqrt(your_model.deg_rate_std**2 + opponent_new_model.deg_rate_std**2)
        * n_laps_to_simulate
    )

    ci_lower = net_gain - 1.96 * uncertainty
    ci_upper = net_gain + 1.96 * uncertainty

    logger.info(
        f"Overcut analysis opponent pit lap {opponent_pit_lap}: gain={net_gain:.2f}s "
        f"(opp_pit_loss={pit_loss:.1f}s, opp_warmup={total_opponent_warmup:.2f}s)"
    )

    return UndercutAnalysis(
        pit_lap=opponent_pit_lap,
        your_compound=your_compound,
        opponent_compound=opponent_new_compound,
        estimated_gain=float(net_gain),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        breakdown=breakdown,
    )


def create_undercut_heatmap(
    race_laps: int,
    your_compound: str,
    your_stint_age_start: int,
    new_compound: str,
    opponent_compound: str,
    opponent_stint_age_start: int,
    degradation_models: dict[str, DegradationModel],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> go.Figure:
    """Create heatmap showing undercut gain vs pit lap.

    Visualization for race engineers to identify optimal undercut window.

    Args:
        race_laps: Total race laps
        your_compound: Your current compound
        your_stint_age_start: Your starting stint age
        new_compound: Compound you'd switch to
        opponent_compound: Opponent's compound
        opponent_stint_age_start: Opponent's starting stint age
        degradation_models: Degradation models
        config: Strategy configuration

    Returns:
        Plotly heatmap figure
    """
    rng = np.random.default_rng(config.random_seed)

    # Analyze pit laps from current to near end
    min_pit_lap = config.min_stint_length + 1
    max_pit_lap = race_laps - config.min_stint_length

    pit_laps = []
    gains = []
    ci_lowers = []
    ci_uppers = []

    for pit_lap in range(min_pit_lap, max_pit_lap + 1):
        # Calculate current stint ages at this lap
        your_age = your_stint_age_start + pit_lap - 1
        opponent_age = opponent_stint_age_start + pit_lap - 1

        try:
            analysis = estimate_undercut_gain(
                current_lap=1,  # Relative calculation
                pit_lap=pit_lap,
                your_current_compound=your_compound,
                your_new_compound=new_compound,
                opponent_compound=opponent_compound,
                your_stint_age=your_age,
                opponent_stint_age=opponent_age,
                degradation_models=degradation_models,
                config=config,
                rng=rng,
            )

            pit_laps.append(pit_lap)
            gains.append(analysis.estimated_gain)
            ci_lowers.append(analysis.confidence_interval[0])
            ci_uppers.append(analysis.confidence_interval[1])

        except Exception as e:
            logger.warning(f"Failed to analyze lap {pit_lap}: {e}")
            continue

    # Create figure
    fig = go.Figure()

    # Main line
    fig.add_trace(
        go.Scatter(
            x=pit_laps,
            y=gains,
            mode="lines+markers",
            name="Expected Gain",
            line=dict(width=3, color="#FF1E1E"),
            marker=dict(size=8),
        )
    )

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=pit_laps + pit_laps[::-1],
            y=ci_uppers + ci_lowers[::-1],
            fill="toself",
            fillcolor="rgba(255,30,30,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="95% CI",
        )
    )

    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        annotation_text="Break-even",
    )

    # Mark optimal lap
    if gains:
        optimal_idx = np.argmax(gains)
        optimal_lap = pit_laps[optimal_idx]
        optimal_gain = gains[optimal_idx]

        fig.add_vline(
            x=optimal_lap,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Optimal: Lap {optimal_lap} (+{optimal_gain:.2f}s)",
            annotation_position="top",
        )

    fig.update_layout(
        title=f"Undercut Analysis: {your_compound} → {new_compound} vs {opponent_compound}",
        xaxis_title="Pit Lap",
        yaxis_title="Expected Gain (seconds)",
        template=config.plot_theme,
        width=config.plot_width,
        height=config.plot_height,
        hovermode="x unified",
    )

    return fig


def find_optimal_undercut_window(
    race_laps: int,
    your_compound: str,
    your_stint_age_start: int,
    new_compound: str,
    opponent_compound: str,
    opponent_stint_age_start: int,
    degradation_models: dict[str, DegradationModel],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> PitWindowRecommendation:
    """Find optimal undercut window with risk assessment.

    Args:
        race_laps: Total race laps
        your_compound: Your current compound
        your_stint_age_start: Your starting stint age
        new_compound: Compound you'd switch to
        opponent_compound: Opponent's compound
        opponent_stint_age_start: Opponent's starting stint age
        degradation_models: Degradation models
        config: Strategy configuration

    Returns:
        PitWindowRecommendation with optimal timing
    """
    rng = np.random.default_rng(config.random_seed)

    min_pit_lap = config.min_stint_length + 1
    max_pit_lap = race_laps - config.min_stint_length

    gains = []
    uncertainties = []

    for pit_lap in range(min_pit_lap, max_pit_lap + 1):
        your_age = your_stint_age_start + pit_lap - 1
        opponent_age = opponent_stint_age_start + pit_lap - 1

        try:
            analysis = estimate_undercut_gain(
                current_lap=1,
                pit_lap=pit_lap,
                your_current_compound=your_compound,
                your_new_compound=new_compound,
                opponent_compound=opponent_compound,
                your_stint_age=your_age,
                opponent_stint_age=opponent_age,
                degradation_models=degradation_models,
                config=config,
                rng=rng,
            )

            gains.append(analysis.estimated_gain)
            uncertainty = (analysis.confidence_interval[1] - analysis.confidence_interval[0]) / 2
            uncertainties.append(uncertainty)

        except Exception:
            gains.append(-999)
            uncertainties.append(999)

    gains = np.array(gains)
    uncertainties = np.array(uncertainties)

    # Find optimal lap
    optimal_idx = np.argmax(gains)
    optimal_lap = min_pit_lap + optimal_idx
    expected_gain = gains[optimal_idx]

    # Find window (laps with positive gain)
    positive_mask = gains > 0
    if np.any(positive_mask):
        positive_laps = np.where(positive_mask)[0]
        window_start = min_pit_lap + positive_laps[0]
        window_end = min_pit_lap + positive_laps[-1]
    else:
        window_start = optimal_lap
        window_end = optimal_lap

    # Risk assessment based on uncertainty
    avg_uncertainty = uncertainties[optimal_idx]
    if avg_uncertainty < 0.5:
        risk = "low"
    elif avg_uncertainty < 1.0:
        risk = "medium"
    else:
        risk = "high"

    logger.info(
        f"Optimal undercut window: Lap {optimal_lap} " f"(+{expected_gain:.2f}s, risk={risk})"
    )

    return PitWindowRecommendation(
        optimal_lap=optimal_lap,
        expected_gain=float(expected_gain),
        window_start=window_start,
        window_end=window_end,
        risk_assessment=risk,
    )
