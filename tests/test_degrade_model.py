"""Tests for degradation modeling.

Author: João Pedro Cunha
"""

import numpy as np
import pandas as pd
import pytest

from f1strategy.degrade_model import fit_degradation_model, DegradationModel
from f1strategy.config import StrategyConfig


def create_mock_stint_data(compound: str = "SOFT", n_laps: int = 20):
    """Create synthetic stint data."""
    stint_ages = np.arange(1, n_laps + 1)
    base_time = 90.0
    deg_rate = 0.05

    lap_times = base_time + deg_rate * stint_ages + np.random.normal(0, 0.2, n_laps)

    return pd.DataFrame({
        'Stint': [1] * n_laps,
        'Compound': [compound] * n_laps,
        'LapNumber': np.arange(1, n_laps + 1),
        'LapTime': lap_times,
        'StintAge': stint_ages,
        'IsOutlier': [False] * n_laps,
    })


class TestDegradationModel:
    """Tests for degradation model fitting."""

    def test_fit_linear_model(self):
        """Test fitting a linear degradation model."""
        data = create_mock_stint_data()
        config = StrategyConfig(degradation_model_type="linear")

        model = fit_degradation_model(data, "SOFT", config)

        assert model is not None
        assert model.compound == "SOFT"
        assert model.model_type == "linear"
        assert model.baseline_laptime > 0
        assert model.deg_rate > 0
        assert 0 <= model.r_squared <= 1

    def test_prediction_increases_with_age(self):
        """Test that predicted lap time increases with stint age."""
        data = create_mock_stint_data()
        config = StrategyConfig()

        model = fit_degradation_model(data, "SOFT", config)

        pred_1 = model.predict(1)
        pred_10 = model.predict(10)
        pred_20 = model.predict(20)

        assert pred_10 > pred_1
        assert pred_20 > pred_10

    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None."""
        data = create_mock_stint_data(n_laps=2)  # Too few laps
        config = StrategyConfig()

        model = fit_degradation_model(data, "SOFT", config)

        assert model is None
