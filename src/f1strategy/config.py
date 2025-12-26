"""Configuration module for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

SessionType = Literal["R"]  # Race only for strategy simulation
DegradationModel = Literal["linear", "quadratic", "piecewise"]


@dataclass
class StrategyConfig:
    """Configuration for strategy simulation."""

    # Cache settings
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    enable_cache: bool = True

    # Degradation model settings
    degradation_model_type: DegradationModel = "linear"
    min_stint_length: int = 5  # minimum laps per stint
    outlier_threshold: float = 3.0  # std devs for outlier rejection

    # Monte Carlo settings
    n_simulations: int = 1000
    random_seed: int = 42

    # Pit stop settings
    pit_loss_mean: float = 22.0  # seconds
    pit_loss_std: float = 1.5  # seconds

    # Safety car settings
    safety_car_prob: float = 0.3  # probability of SC in race
    sc_lap_time_reduction: float = 10.0  # seconds per lap under SC
    sc_duration_mean: int = 5  # laps
    sc_duration_std: int = 2  # laps
    sc_pit_advantage: float = 15.0  # extra seconds saved if pit under SC

    # Strategy constraints
    max_stops: int = 2
    require_two_compounds: bool = False  # may not have data for all races

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    plot_dpi: int = 150
    plot_width: int = 1200
    plot_height: int = 600
    plot_theme: str = "plotly_dark"

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.n_simulations < 100:
            raise ValueError("n_simulations must be at least 100")
        if self.min_stint_length < 1:
            raise ValueError("min_stint_length must be positive")
        if self.pit_loss_mean < 0:
            raise ValueError("pit_loss_mean cannot be negative")

        logger.info("Configuration initialized successfully")


DEFAULT_CONFIG = StrategyConfig()
