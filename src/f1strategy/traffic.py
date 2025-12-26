"""Traffic penalty modeling for F1 Race Strategy Simulator.

Simplified traffic proxy model for estimating rejoin penalties. Does not simulate
full grid positions (too complex), but provides reasonable traffic cost estimates
based on expected rejoin gap and track characteristics.

Physics reasoning: Traffic causes lap time loss through:
1. Dirty air reducing downforce
2. Blocked overtaking opportunities on certain track sections
3. Compromised racing lines

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass

import pandas as pd

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig, TrackType
from f1strategy.simulator import Strategy

logger = logging.getLogger(__name__)


@dataclass
class TrafficParameters:
    """Parameters for traffic penalty calculation."""

    track_type: TrackType
    expected_rejoin_gap: float  # seconds behind leader on rejoin
    typical_gap_between_cars: float = 1.5  # seconds per position
    traffic_duration_laps: int = 3  # laps typically spent in traffic


def estimate_traffic_penalty(
    rejoin_gap: float,
    track_type: TrackType = "permanent",
    config: StrategyConfig = DEFAULT_CONFIG,
) -> float:
    """Estimate traffic penalty based on rejoin gap and track type.

    Strategy reasoning: After pitting, cars rejoin the field. If the rejoin gap
    is small (< 10s), the car will likely emerge in traffic, losing time due to
    dirty air and blocked overtaking.

    Track dependency:
    - Street circuits: High penalty (limited overtaking, dirty air critical)
    - Permanent circuits: Medium penalty (some overtaking zones)
    - Hybrid: Between street and permanent

    Args:
        rejoin_gap: Expected gap to cars ahead after pit stop (seconds)
        track_type: Type of circuit
        config: Strategy configuration

    Returns:
        Expected traffic penalty in seconds per lap
    """
    # No traffic if rejoin gap is large
    if rejoin_gap > 10.0:
        return 0.0

    # Base penalty
    base_penalty = config.base_traffic_penalty

    # Track multiplier
    if track_type == "street":
        multiplier = config.traffic_penalty_street
    elif track_type == "permanent":
        multiplier = config.traffic_penalty_permanent
    else:  # hybrid
        multiplier = (config.traffic_penalty_street + config.traffic_penalty_permanent) / 2

    # Gap factor: smaller gap = more traffic
    # Linear interpolation: 0s gap = full penalty, 10s gap = no penalty
    gap_factor = max(0.0, 1.0 - rejoin_gap / 10.0)

    penalty = base_penalty * multiplier * gap_factor

    logger.debug(
        f"Traffic penalty: gap={rejoin_gap:.1f}s, track={track_type}, "
        f"penalty={penalty:.3f}s/lap"
    )

    return float(penalty)


def apply_traffic_to_strategy(
    strategy: Strategy,
    lap_times: list[float],
    traffic_params: TrafficParameters,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> list[float]:
    """Apply traffic penalties to lap times for a strategy.

    Applies traffic penalty to laps immediately following pit stops, based on
    expected rejoin position.

    Args:
        strategy: Racing strategy
        lap_times: Predicted lap times without traffic
        traffic_params: Traffic parameters
        config: Strategy configuration

    Returns:
        Adjusted lap times with traffic penalties
    """
    if not config.enable_traffic_model:
        return lap_times

    adjusted_times = lap_times.copy()

    # Identify laps after pit stops
    for stint_idx, stint in enumerate(strategy.stints):
        if stint_idx == 0:
            # First stint: no pit stop before it
            continue

        # Laps immediately after pit stop
        pit_lap_idx = stint.start_lap - 1  # Convert to 0-indexed
        traffic_end_lap = min(
            pit_lap_idx + traffic_params.traffic_duration_laps,
            len(lap_times),
        )

        # Calculate traffic penalty
        penalty_per_lap = estimate_traffic_penalty(
            traffic_params.expected_rejoin_gap,
            traffic_params.track_type,
            config,
        )

        # Apply penalty with decay (most traffic on first lap after rejoin)
        for lap_offset in range(traffic_params.traffic_duration_laps):
            lap_idx = pit_lap_idx + lap_offset
            if lap_idx >= len(adjusted_times):
                break

            # Decay factor: first lap has full penalty, then reduces
            decay = 1.0 - (lap_offset / traffic_params.traffic_duration_laps)
            adjusted_times[lap_idx] += penalty_per_lap * decay

    total_penalty = sum(adjusted_times) - sum(lap_times)
    logger.info(f"Total traffic penalty applied: {total_penalty:.2f}s")

    return adjusted_times


def estimate_rejoin_gap(
    current_position: int,
    pit_loss: float,
    typical_lap_time: float,
    typical_gap_per_position: float = 1.5,
) -> float:
    """Estimate expected gap to field after pit stop.

    Simple estimation based on current position and pit loss.

    Args:
        current_position: Current race position (1 = leader)
        pit_loss: Pit stop time loss (seconds)
        typical_lap_time: Typical lap time (seconds)
        typical_gap_per_position: Average gap between positions (seconds)

    Returns:
        Estimated gap to car ahead after rejoining (seconds)
    """
    if current_position == 1:
        # Leader: gap is determined by pit loss vs second place
        gap = pit_loss - typical_gap_per_position
    else:
        # Mid-pack: lose time from pit stop minus what field gained
        field_progress = typical_lap_time
        net_loss = pit_loss - (typical_gap_per_position * 0.5)  # Simplified
        gap = net_loss

    return max(0.0, float(gap))


def create_traffic_sensitivity_analysis(
    strategy: Strategy,
    base_lap_times: list[float],
    track_type: TrackType,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Analyze sensitivity to different rejoin gap scenarios.

    Creates a table showing how total race time changes with different
    traffic scenarios.

    Args:
        strategy: Racing strategy
        base_lap_times: Base lap times without traffic
        track_type: Track type
        config: Strategy configuration

    Returns:
        DataFrame with sensitivity analysis
    """
    scenarios = []

    rejoin_gaps = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]  # seconds

    base_total = sum(base_lap_times)

    for gap in rejoin_gaps:
        traffic_params = TrafficParameters(
            track_type=track_type,
            expected_rejoin_gap=gap,
        )

        adjusted_times = apply_traffic_to_strategy(
            strategy,
            base_lap_times,
            traffic_params,
            config,
        )

        total_time = sum(adjusted_times)
        penalty = total_time - base_total

        scenarios.append(
            {
                "Rejoin Gap (s)": gap,
                "Total Time (s)": f"{total_time:.2f}",
                "Traffic Penalty (s)": f"{penalty:.2f}",
                "Laps in Traffic": traffic_params.traffic_duration_laps,
            }
        )

    return pd.DataFrame(scenarios)


def estimate_overtaking_probability(
    pace_advantage: float,
    track_type: TrackType,
    laps_available: int,
) -> float:
    """Estimate probability of completing an overtake.

    Simplified model for overtaking probability based on pace advantage
    and track characteristics.

    Args:
        pace_advantage: Seconds per lap faster than car ahead
        track_type: Track type (affects overtaking difficulty)
        laps_available: Number of laps available to overtake

    Returns:
        Probability of successful overtake (0-1)
    """
    # Base probability per lap attempt
    if track_type == "street":
        base_prob_per_lap = 0.05  # Very hard to overtake
    elif track_type == "permanent":
        base_prob_per_lap = 0.15  # Easier with DRS zones
    else:  # hybrid
        base_prob_per_lap = 0.10

    # Scale by pace advantage (need ~0.5s advantage minimum)
    if pace_advantage < 0.3:
        pace_factor = 0.1
    else:
        pace_factor = min(1.0, pace_advantage / 0.5)

    prob_per_lap = base_prob_per_lap * pace_factor

    # Probability of overtaking within N laps
    prob_overall = 1.0 - (1.0 - prob_per_lap) ** laps_available

    return float(prob_overall)
