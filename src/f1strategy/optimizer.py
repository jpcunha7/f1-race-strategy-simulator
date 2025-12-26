"""Strategy optimization for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import logging
from typing import Optional

import numpy as np

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1strategy.degrade_model import DegradationModel
from f1strategy.simulator import Strategy, Stint, run_monte_carlo, SimulationResult

logger = logging.getLogger(__name__)


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
