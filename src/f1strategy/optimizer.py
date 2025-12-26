"""Strategy optimization for F1 Race Strategy Simulator.

Enhanced with professional race strategy analysis:
- Strategy dominance probability (A beats B X% of time)
- Risk profiles (best-case/worst-case percentiles)
- Sensitivity analysis (which parameters matter most)
- Executive summaries for decision makers

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1strategy.degrade_model import DegradationModel
from f1strategy.simulator import Strategy, Stint, run_monte_carlo, SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class StrategyRiskProfile:
    """Risk profile for a strategy."""

    strategy_name: str
    mean_time: float
    median_time: float
    std_time: float
    best_case: float  # 5th percentile (optimistic)
    worst_case: float  # 95th percentile (pessimistic)
    probability_top_strategy: float  # probability of being fastest


@dataclass
class StrategyComparison:
    """Head-to-head comparison between two strategies."""

    strategy_a: str
    strategy_b: str
    prob_a_beats_b: float  # probability A is faster than B
    mean_delta: float  # mean time difference (A - B)
    median_delta: float


@dataclass
class SensitivityResult:
    """Sensitivity analysis result."""

    parameter_name: str
    base_value: float
    varied_value: float
    mean_time_change: float  # seconds
    rank_change: int  # change in strategy ranking


def generate_one_stop_strategies(
    total_laps: int,
    available_compounds: list[str],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> list[Strategy]:
    """Generate all valid 1-stop strategies."""
    strategies = []

    # Try different pit windows
    min_stint = config.min_stint_length
    max_first_stint = total_laps - min_stint

    for pit_lap in range(min_stint, max_first_stint + 1):
        # Try all compound combinations
        for compound1 in available_compounds:
            for compound2 in available_compounds:
                if config.require_two_compounds and compound1 == compound2:
                    continue

                stints = [
                    Stint(compound1, 1, pit_lap),
                    Stint(compound2, pit_lap + 1, total_laps),
                ]

                strategy = Strategy(
                    stints=stints, description=f"1-stop: {compound1} -> {compound2} (Lap {pit_lap})"
                )

                if strategy.validate(total_laps, min_stint):
                    strategies.append(strategy)

    logger.info(f"Generated {len(strategies)} valid 1-stop strategies")
    return strategies


def generate_two_stop_strategies(
    total_laps: int,
    available_compounds: list[str],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> list[Strategy]:
    """Generate all valid 2-stop strategies."""
    strategies = []

    min_stint = config.min_stint_length

    # Sample pit windows to avoid explosion
    pit1_options = range(min_stint, total_laps - 2 * min_stint + 1, 3)
    pit2_options_base = range(min_stint, total_laps - min_stint + 1, 3)

    for pit1 in pit1_options:
        for pit2 in pit2_options_base:
            if pit2 <= pit1 + min_stint:
                continue
            if pit2 > total_laps - min_stint:
                continue

            for c1 in available_compounds:
                for c2 in available_compounds:
                    for c3 in available_compounds:
                        stints = [
                            Stint(c1, 1, pit1),
                            Stint(c2, pit1 + 1, pit2),
                            Stint(c3, pit2 + 1, total_laps),
                        ]

                        strategy = Strategy(
                            stints=stints, description=f"2-stop: {c1}->{c2}->{c3} (L{pit1},L{pit2})"
                        )

                        if strategy.validate(total_laps, min_stint):
                            strategies.append(strategy)

    logger.info(f"Generated {len(strategies)} valid 2-stop strategies")
    return strategies


def optimize_strategy(
    degradation_models: dict[str, DegradationModel],
    total_laps: int,
    config: StrategyConfig = DEFAULT_CONFIG,
    max_strategies_to_test: int = 50,
) -> tuple[list[Strategy], dict[str, list[SimulationResult]]]:
    """Find optimal strategy through grid search."""
    available_compounds = list(degradation_models.keys())

    if not available_compounds:
        raise ValueError("No degradation models available")

    logger.info(f"Optimizing for {total_laps} laps with compounds: {available_compounds}")

    # Generate candidate strategies
    one_stop_strategies = generate_one_stop_strategies(total_laps, available_compounds, config)
    two_stop_strategies = []

    if config.max_stops >= 2:
        two_stop_strategies = generate_two_stop_strategies(total_laps, available_compounds, config)

    all_strategies = one_stop_strategies + two_stop_strategies

    # Limit strategies to test
    if len(all_strategies) > max_strategies_to_test:
        logger.warning(
            f"Too many strategies ({len(all_strategies)}), " f"sampling {max_strategies_to_test}"
        )
        # Sample uniformly
        indices = np.linspace(0, len(all_strategies) - 1, max_strategies_to_test, dtype=int)
        all_strategies = [all_strategies[i] for i in indices]

    # Run simulations for each strategy
    results_dict = {}

    for strategy in all_strategies:
        results = run_monte_carlo(
            strategy,
            degradation_models,
            total_laps,
            config,
            show_progress=False,
        )
        results_dict[strategy.description] = results

    # Rank by mean time
    mean_times = {
        name: np.mean([r.total_time for r in results]) for name, results in results_dict.items()
    }

    ranked_strategies = sorted(all_strategies, key=lambda s: mean_times[s.description])

    logger.info(
        f"Best strategy: {ranked_strategies[0].description} "
        f"(mean time: {mean_times[ranked_strategies[0].description]:.1f}s)"
    )

    return ranked_strategies, results_dict


def analyze_pit_window(
    compound1: str,
    compound2: str,
    total_laps: int,
    degradation_models: dict[str, DegradationModel],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> dict:
    """Analyze optimal pit window for a specific compound combination."""
    min_stint = config.min_stint_length
    pit_laps = range(min_stint, total_laps - min_stint + 1)

    pit_analysis = []

    for pit_lap in pit_laps:
        stints = [
            Stint(compound1, 1, pit_lap),
            Stint(compound2, pit_lap + 1, total_laps),
        ]

        strategy = Strategy(stints=stints, description=f"{compound1}->{compound2} @ L{pit_lap}")

        if not strategy.validate(total_laps, min_stint):
            continue

        # Run fewer simulations for pit window analysis
        quick_config = StrategyConfig(**{**config.__dict__, "n_simulations": 200})

        results = run_monte_carlo(
            strategy,
            degradation_models,
            total_laps,
            quick_config,
            show_progress=False,
        )

        times = [r.total_time for r in results]

        pit_analysis.append(
            {
                "pit_lap": pit_lap,
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "median_time": np.median(times),
            }
        )

    return {
        "compound1": compound1,
        "compound2": compound2,
        "analysis": pit_analysis,
    }


def calculate_strategy_dominance(
    strategy_a_results: list[SimulationResult],
    strategy_b_results: list[SimulationResult],
) -> StrategyComparison:
    """Calculate probability that strategy A beats strategy B.

    Strategy reasoning: Race engineers need to know not just average performance,
    but the probability of one strategy beating another across all scenarios
    (with/without safety cars, different degradation realizations, etc.).

    Args:
        strategy_a_results: Simulation results for strategy A
        strategy_b_results: Simulation results for strategy B

    Returns:
        StrategyComparison with dominance probability
    """
    times_a = np.array([r.total_time for r in strategy_a_results])
    times_b = np.array([r.total_time for r in strategy_b_results])

    # Monte Carlo: sample pairs and count wins
    n_comparisons = min(len(times_a), len(times_b))
    wins_a = np.sum(times_a[:n_comparisons] < times_b[:n_comparisons])
    prob_a_beats_b = wins_a / n_comparisons

    mean_delta = np.mean(times_a) - np.mean(times_b)
    median_delta = np.median(times_a) - np.median(times_b)

    strategy_a_name = strategy_a_results[0].strategy.description
    strategy_b_name = strategy_b_results[0].strategy.description

    return StrategyComparison(
        strategy_a=strategy_a_name,
        strategy_b=strategy_b_name,
        prob_a_beats_b=float(prob_a_beats_b),
        mean_delta=float(mean_delta),
        median_delta=float(median_delta),
    )


def calculate_risk_profiles(
    results_dict: dict[str, list[SimulationResult]],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> dict[str, StrategyRiskProfile]:
    """Calculate risk profiles for all strategies.

    Risk profile includes:
    - Best case (5th percentile): optimistic scenario
    - Worst case (95th percentile): pessimistic scenario
    - Probability of being fastest overall

    Args:
        results_dict: Dictionary of strategy results
        config: Strategy configuration

    Returns:
        Dictionary mapping strategy name to risk profile
    """
    risk_profiles = {}

    # Collect all times for cross-strategy comparison
    all_strategy_times = {}
    for name, results in results_dict.items():
        all_strategy_times[name] = [r.total_time for r in results]

    # Calculate probability each strategy is fastest
    n_sims = len(list(all_strategy_times.values())[0])
    fastest_counts = {name: 0 for name in results_dict.keys()}

    for sim_idx in range(n_sims):
        # Get time for each strategy in this simulation
        sim_times = {name: times[sim_idx] for name, times in all_strategy_times.items()}
        fastest_strategy = min(sim_times, key=sim_times.get)
        fastest_counts[fastest_strategy] += 1

    # Calculate risk profiles
    for name, results in results_dict.items():
        times = all_strategy_times[name]
        p_low, p_high = config.risk_percentiles

        risk_profiles[name] = StrategyRiskProfile(
            strategy_name=name,
            mean_time=float(np.mean(times)),
            median_time=float(np.median(times)),
            std_time=float(np.std(times)),
            best_case=float(np.percentile(times, p_low)),
            worst_case=float(np.percentile(times, p_high)),
            probability_top_strategy=float(fastest_counts[name] / n_sims),
        )

    return risk_profiles


def create_strategy_executive_summary(
    results_dict: dict[str, list[SimulationResult]],
    risk_profiles: dict[str, StrategyRiskProfile],
    top_n: int = 3,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> str:
    """Create executive summary for strategy decision.

    Format suitable for race engineers and team principals.

    Args:
        results_dict: Strategy results
        risk_profiles: Risk profiles for each strategy
        top_n: Number of top strategies to include
        config: Strategy configuration

    Returns:
        Formatted executive summary string
    """
    # Rank strategies by mean time
    ranked = sorted(
        risk_profiles.items(),
        key=lambda x: x[1].mean_time,
    )

    summary_lines = [
        "=" * 80,
        "STRATEGY EXECUTIVE SUMMARY",
        "=" * 80,
        "",
    ]

    # Top strategies
    summary_lines.append(f"TOP {top_n} RECOMMENDED STRATEGIES:")
    summary_lines.append("-" * 80)

    for rank, (name, profile) in enumerate(ranked[:top_n], 1):
        ci_width = profile.worst_case - profile.best_case
        confidence = "High" if ci_width < 5.0 else "Medium" if ci_width < 10.0 else "Low"

        summary_lines.append(f"\n{rank}. {name}")
        summary_lines.append(
            f"   Expected Time: {profile.mean_time:.2f}s ± {profile.std_time:.2f}s"
        )
        summary_lines.append(
            f"   Best Case: {profile.best_case:.2f}s  |  Worst Case: {profile.worst_case:.2f}s"
        )
        summary_lines.append(
            f"   Probability Fastest: {profile.probability_top_strategy * 100:.1f}%"
        )
        summary_lines.append(f"   Confidence: {confidence}")

        # Time delta to best
        if rank > 1:
            delta = profile.mean_time - ranked[0][1].mean_time
            summary_lines.append(f"   Gap to P1: +{delta:.2f}s")

    # Risk assessment
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("RISK ASSESSMENT:")
    summary_lines.append("-" * 80)

    best_strategy = ranked[0][0]
    best_profile = ranked[0][1]

    summary_lines.append(f"\nRecommended: {best_strategy}")
    summary_lines.append(
        f"Risk Level: "
        f"{'Low' if best_profile.worst_case - best_profile.best_case < 5.0 else 'Medium'}"
    )

    # Head-to-head comparisons
    if len(ranked) > 1:
        summary_lines.append("\nHEAD-TO-HEAD COMPARISONS:")
        for i in range(1, min(3, len(ranked))):
            comparison = calculate_strategy_dominance(
                results_dict[ranked[0][0]],
                results_dict[ranked[i][0]],
            )
            summary_lines.append(
                f"  P1 vs P{i+1}: {comparison.prob_a_beats_b * 100:.1f}% win rate "
                f"(avg delta: {abs(comparison.mean_delta):.2f}s)"
            )

    summary_lines.append("\n" + "=" * 80)

    return "\n".join(summary_lines)


def analyze_sensitivity(
    base_strategy: Strategy,
    degradation_models: dict[str, DegradationModel],
    total_laps: int,
    base_config: StrategyConfig,
    parameter_variations: Optional[dict[str, list[float]]] = None,
) -> dict[str, list[SensitivityResult]]:
    """Analyze sensitivity of strategy performance to parameter changes.

    Tests how strategy ranking changes when key parameters vary.
    Critical for understanding decision robustness.

    Args:
        base_strategy: Strategy to analyze
        degradation_models: Degradation models
        total_laps: Total race laps
        base_config: Base configuration
        parameter_variations: Optional custom parameter variations

    Returns:
        Dictionary mapping parameter name to sensitivity results
    """
    if parameter_variations is None:
        # Default variations
        parameter_variations = {
            "pit_loss_mean": [20.0, 22.0, 24.0],
            "safety_car_prob": [0.0, 0.3, 0.6],
        }

    # Run base case
    base_results = run_monte_carlo(
        base_strategy,
        degradation_models,
        total_laps,
        base_config,
        show_progress=False,
    )
    base_mean = np.mean([r.total_time for r in base_results])

    sensitivity_results = {}

    for param_name, values in parameter_variations.items():
        param_results = []

        for value in values:
            # Create modified config
            config_dict = base_config.__dict__.copy()
            if param_name in config_dict:
                old_value = config_dict[param_name]
                config_dict[param_name] = value

                modified_config = StrategyConfig(**config_dict)

                # Run simulation with modified parameter
                varied_results = run_monte_carlo(
                    base_strategy,
                    degradation_models,
                    total_laps,
                    modified_config,
                    show_progress=False,
                )

                varied_mean = np.mean([r.total_time for r in varied_results])
                delta = varied_mean - base_mean

                param_results.append(
                    SensitivityResult(
                        parameter_name=param_name,
                        base_value=float(old_value) if isinstance(old_value, (int, float)) else 0.0,
                        varied_value=float(value),
                        mean_time_change=float(delta),
                        rank_change=0,  # Would need multiple strategies to assess
                    )
                )

        sensitivity_results[param_name] = param_results

    return sensitivity_results
