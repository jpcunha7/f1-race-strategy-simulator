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
class SafetyCarEvent:
    """Safety car or VSC event."""

    event_type: str  # "SC" or "VSC"
    start_lap: int
    duration: int  # laps


@dataclass
class SimulationResult:
    """Result of a single race simulation."""

    strategy: Strategy
    total_time: float  # seconds
    lap_times: list[float]
    had_safety_car: bool
    sc_start_lap: Optional[int]
    sc_duration: Optional[int]
    safety_events: list[SafetyCarEvent]  # All SC/VSC events


def sample_sc_vsc_events(
    total_laps: int,
    config: StrategyConfig,
    rng: np.random.Generator,
) -> list[SafetyCarEvent]:
    """Sample SC and VSC events for a race.

    Physics reasoning: SC and VSC are independent random events that
    can occur during a race. SC is triggered by crashes/incidents requiring
    track marshals. VSC is for less severe incidents.

    SC vs VSC differences:
    - SC: Full safety car, leads pack, 30% lap time reduction, large pit advantage
    - VSC: Virtual safety car, no physical car, 40% lap time reduction, smaller pit advantage

    Args:
        total_laps: Total race laps
        config: Strategy configuration
        rng: Random number generator

    Returns:
        List of safety car events
    """
    events = []

    # Sample number of SC events
    if rng.random() < config.safety_car_prob:
        n_sc = rng.integers(1, config.max_sc_events + 1)

        for _ in range(n_sc):
            # SC can't happen in first 3 or last 5 laps
            start_lap = rng.integers(4, max(5, total_laps - 5))
            duration = int(rng.normal(config.sc_duration_mean, config.sc_duration_std))
            duration = max(2, min(duration, total_laps - start_lap))

            events.append(
                SafetyCarEvent(
                    event_type="SC",
                    start_lap=start_lap,
                    duration=duration,
                )
            )

    # Sample number of VSC events
    if rng.random() < config.vsc_prob:
        n_vsc = rng.integers(1, config.max_vsc_events + 1)

        for _ in range(n_vsc):
            start_lap = rng.integers(4, max(5, total_laps - 5))
            duration = int(rng.normal(config.vsc_duration_mean, config.vsc_duration_std))
            duration = max(1, min(duration, total_laps - start_lap))

            events.append(
                SafetyCarEvent(
                    event_type="VSC",
                    start_lap=start_lap,
                    duration=duration,
                )
            )

    # Sort by start lap
    events.sort(key=lambda e: e.start_lap)

    # Remove overlapping events (keep first)
    filtered_events = []
    occupied_laps = set()

    for event in events:
        event_laps = set(range(event.start_lap, event.start_lap + event.duration))
        if not event_laps.intersection(occupied_laps):
            filtered_events.append(event)
            occupied_laps.update(event_laps)

    return filtered_events


def simulate_safety_car(
    total_laps: int,
    config: StrategyConfig,
    rng: np.random.Generator,
) -> tuple[bool, Optional[int], Optional[int]]:
    """Simulate if/when a safety car occurs.

    Legacy function for backward compatibility. Use sample_sc_vsc_events for new code.
    """
    events = sample_sc_vsc_events(total_laps, config, rng)
    sc_events = [e for e in events if e.event_type == "SC"]

    if sc_events:
        return True, sc_events[0].start_lap, sc_events[0].duration
    return False, None, None


def simulate_race(
    strategy: Strategy,
    degradation_models: dict[str, DegradationModel],
    total_laps: int,
    config: StrategyConfig = DEFAULT_CONFIG,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:
    """Simulate a single race with given strategy.

    Enhanced with SC/VSC modeling and warmup penalties.
    """
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    lap_times = []
    total_time = 0.0

    # Sample SC/VSC events
    safety_events = sample_sc_vsc_events(total_laps, config, rng)

    # Create lookup for safety event effects by lap
    safety_effects = {}  # lap -> (reduction_factor, pit_advantage)
    for event in safety_events:
        for lap_offset in range(event.duration):
            lap = event.start_lap + lap_offset

            if event.event_type == "SC":
                safety_effects[lap] = (
                    config.sc_lap_time_reduction,
                    config.sc_pit_advantage,
                )
            else:  # VSC
                safety_effects[lap] = (
                    config.vsc_lap_time_reduction,
                    config.vsc_pit_advantage,
                )

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

            # Apply warmup penalty (physics: cold tires = less grip)
            if stint_age <= 3:  # First few laps on new tires
                warmup_penalty = calculate_warmup_penalty(stint_age, config)
                lap_time += warmup_penalty

            # Apply SC/VSC effect
            if lap in safety_effects:
                reduction_pct, _ = safety_effects[lap]
                lap_time *= (1.0 - reduction_pct)

            lap_times.append(lap_time)
            total_time += lap_time

        # Add pit stop time (if not last stint)
        if stint != strategy.stints[-1]:
            pit_loss = rng.normal(config.pit_loss_mean, config.pit_loss_std)

            # Check if pit during SC/VSC
            pit_lap = stint.end_lap
            if pit_lap in safety_effects:
                _, pit_advantage = safety_effects[pit_lap]
                pit_loss -= pit_advantage

            total_time += pit_loss

    # Legacy compatibility
    sc_events = [e for e in safety_events if e.event_type == "SC"]
    had_sc = len(sc_events) > 0
    sc_start = sc_events[0].start_lap if sc_events else None
    sc_duration = sc_events[0].duration if sc_events else None

    return SimulationResult(
        strategy=strategy,
        total_time=total_time,
        lap_times=lap_times,
        had_safety_car=had_sc,
        sc_start_lap=sc_start,
        sc_duration=sc_duration,
        safety_events=safety_events,
    )


def calculate_warmup_penalty(
    stint_age: int,
    config: StrategyConfig,
) -> float:
    """Calculate tire warmup penalty for given stint age.

    Imported from undercut module logic for use in race simulation.

    Args:
        stint_age: Lap number within stint
        config: Strategy configuration

    Returns:
        Warmup penalty in seconds
    """
    if stint_age < 1:
        return 0.0

    if config.warmup_model == "exponential":
        penalty = config.warmup_penalty_initial * np.exp(
            -stint_age / config.warmup_decay_tau
        )
        return float(penalty)
    else:
        # Step function
        if stint_age <= config.warmup_step_laps:
            return config.warmup_penalty_initial
        else:
            return 0.0


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
