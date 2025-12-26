"""Race simulation with Monte Carlo for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1strategy.degrade_model import DegradationModel

logger = logging.getLogger(__name__)


@dataclass
class Stint:
    """Represents a racing stint."""

    compound: str
    start_lap: int
    end_lap: int

    @property
    def length(self) -> int:
        return self.end_lap - self.start_lap + 1


@dataclass
class Strategy:
    """Represents a pit strategy."""

    stints: list[Stint]
    description: str = ""

    @property
    def num_stops(self) -> int:
        return len(self.stints) - 1

    def validate(self, total_laps: int, min_stint_length: int) -> bool:
        """Validate strategy meets constraints."""
        # Check total laps
        total = sum(s.length for s in self.stints)
        if total != total_laps:
            return False

        # Check stint lengths
        for stint in self.stints:
            if stint.length < min_stint_length:
                return False

        # Check continuity
        for i in range(len(self.stints) - 1):
            if self.stints[i].end_lap + 1 != self.stints[i + 1].start_lap:
                return False

        return True


@dataclass
class SimulationResult:
    """Result of a single race simulation."""

    strategy: Strategy
    total_time: float  # seconds
    lap_times: list[float]
    had_safety_car: bool
    sc_start_lap: Optional[int]
    sc_duration: Optional[int]


def simulate_safety_car(
    total_laps: int,
    config: StrategyConfig,
    rng: np.random.Generator,
) -> tuple[bool, Optional[int], Optional[int]]:
    """Simulate if/when a safety car occurs."""
    if rng.random() < config.safety_car_prob:
        # SC occurs - pick random lap (not first 3 or last 5)
        sc_start = rng.integers(4, max(5, total_laps - 5))
        sc_duration = int(rng.normal(config.sc_duration_mean, config.sc_duration_std))
        sc_duration = max(2, min(sc_duration, total_laps - sc_start))
        return True, sc_start, sc_duration
    return False, None, None


def simulate_race(
    strategy: Strategy,
    degradation_models: dict[str, DegradationModel],
    total_laps: int,
    config: StrategyConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:
    """Simulate a single race with given strategy."""
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    lap_times = []
    total_time = 0.0

    # Simulate safety car
    had_sc, sc_start, sc_duration = simulate_safety_car(total_laps, config, rng)
    sc_laps = set()
    if had_sc and sc_start is not None and sc_duration is not None:
        sc_laps = set(range(sc_start, sc_start + sc_duration))

    # Simulate each stint
    for stint in strategy.stints:
        model = degradation_models.get(stint.compound)

        if model is None:
            # Fallback if compound not modeled
            logger.warning(f"No model for {stint.compound}, using default")
            base_time = 90.0
            deg_rate = 0.05
        else:
            base_time = model.baseline_laptime
            deg_rate = model.deg_rate

        stint_age = 0
        for lap in range(stint.start_lap, stint.end_lap + 1):
            stint_age += 1

            # Get base lap time with degradation
            if model is not None:
                lap_time = model.predict_with_noise(stint_age, rng)
            else:
                lap_time = base_time + deg_rate * stint_age + rng.normal(0, 0.2)

            # Apply SC effect
            if lap in sc_laps:
                lap_time -= config.sc_lap_time_reduction

            lap_times.append(lap_time)
            total_time += lap_time

        # Add pit stop time (if not last stint)
        if stint != strategy.stints[-1]:
            pit_loss = rng.normal(config.pit_loss_mean, config.pit_loss_std)

            # Check if pit during SC
            pit_lap = stint.end_lap
            if pit_lap in sc_laps:
                pit_loss -= config.sc_pit_advantage

            total_time += pit_loss

    return SimulationResult(
        strategy=strategy,
        total_time=total_time,
        lap_times=lap_times,
        had_safety_car=had_sc,
        sc_start_lap=sc_start,
        sc_duration=sc_duration,
    )


def run_monte_carlo(
    strategy: Strategy,
    degradation_models: dict[str, DegradationModel],
    total_laps: int,
    config: StrategyConfig = DEFAULT_CONFIG,
    show_progress: bool = True,
) -> list[SimulationResult]:
    """Run Monte Carlo simulation for a strategy."""
    results = []

    iterator = range(config.n_simulations)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Simulating {strategy.description}")

    for i in iterator:
        # Create unique RNG for each simulation
        sim_rng = np.random.default_rng(config.random_seed + i)
        result = simulate_race(strategy, degradation_models, total_laps, config, sim_rng)
        results.append(result)

    logger.info(
        f"Completed {len(results)} simulations for {strategy.description}: "
        f"mean time = {np.mean([r.total_time for r in results]):.1f}s"
    )

    return results


def compare_strategies(results_dict: dict[str, list[SimulationResult]]) -> pd.DataFrame:
    """Compare multiple strategies statistically."""
    comparison_data = []

    for strategy_name, results in results_dict.items():
        times = [r.total_time for r in results]

        comparison_data.append(
            {
                "Strategy": strategy_name,
                "Mean Time (s)": np.mean(times),
                "Std Time (s)": np.std(times),
                "Min Time (s)": np.min(times),
                "P25 Time (s)": np.percentile(times, 25),
                "Median Time (s)": np.median(times),
                "P75 Time (s)": np.percentile(times, 75),
                "Max Time (s)": np.max(times),
                "SC Rate": np.mean([r.had_safety_car for r in results]),
            }
        )

    return pd.DataFrame(comparison_data)
