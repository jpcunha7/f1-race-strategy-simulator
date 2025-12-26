"""Tests for race simulator.

Author: João Pedro Cunha
"""

import numpy as np
import pytest

from f1strategy.simulator import Stint, Strategy, simulate_race
from f1strategy.degrade_model import DegradationModel
from f1strategy.config import StrategyConfig


def create_mock_model(compound: str = "SOFT"):
    """Create a mock degradation model."""
    return DegradationModel(
        compound=compound,
        model_type="linear",
        baseline_laptime=90.0,
        deg_rate=0.05,
        deg_rate_std=0.01,
        r_squared=0.95,
        n_samples=100,
        coefficients={"intercept": 90.0, "slope": 0.05},
    )


class TestStrategy:
    """Tests for Strategy class."""

    def test_strategy_validation(self):
        """Test strategy validation."""
        stints = [
            Stint("SOFT", 1, 25),
            Stint("MEDIUM", 26, 50),
        ]
        strategy = Strategy(stints)

        assert strategy.validate(50, 5) is True
        assert strategy.num_stops == 1

    def test_invalid_total_laps(self):
        """Test validation fails for wrong total laps."""
        stints = [
            Stint("SOFT", 1, 25),
            Stint("MEDIUM", 26, 48),  # Only 48 laps total
        ]
        strategy = Strategy(stints)

        assert strategy.validate(50, 5) is False


class TestSimulation:
    """Tests for race simulation."""

    def test_simulation_reproducibility(self):
        """Test that simulation is reproducible with seed."""
        stints = [
            Stint("SOFT", 1, 25),
            Stint("MEDIUM", 26, 50),
        ]
        strategy = Strategy(stints)

        models = {
            "SOFT": create_mock_model("SOFT"),
            "MEDIUM": create_mock_model("MEDIUM"),
        }

        config = StrategyConfig(random_seed=42)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = simulate_race(strategy, models, 50, config, rng1)
        result2 = simulate_race(strategy, models, 50, config, rng2)

        assert abs(result1.total_time - result2.total_time) < 0.001

    def test_simulation_returns_correct_laps(self):
        """Test simulation returns correct number of lap times."""
        stints = [Stint("SOFT", 1, 50)]
        strategy = Strategy(stints)
        models = {"SOFT": create_mock_model()}
        config = StrategyConfig()

        result = simulate_race(strategy, models, 50, config)

        assert len(result.lap_times) == 50
        assert result.total_time > 0
