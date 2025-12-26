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
SafetyCarType = Literal["SC", "VSC"]
TrackType = Literal["street", "permanent", "hybrid"]


@dataclass
class StrategyConfig:
    """Configuration for strategy simulation.

    Professional strategy analysis tool configuration with physics-based modeling,
    validation capabilities, and advanced undercut/traffic analysis.
    """

    # Cache settings
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    enable_cache: bool = True

    # Degradation model settings
    degradation_model_type: DegradationModel = "linear"
    min_stint_length: int = 5  # minimum laps per stint
    outlier_threshold: float = 3.0  # std devs for outlier rejection
    auto_select_degradation_model: bool = True  # use cross-validation to select best model
    cv_folds: int = 3  # cross-validation folds for model selection

    # Monte Carlo settings
    n_simulations: int = 1000
    random_seed: int = 42

    # Pit stop settings
    pit_loss_mean: float = 22.0  # seconds
    pit_loss_std: float = 1.5  # seconds

    # Tire warmup settings (critical for undercut analysis)
    warmup_penalty_initial: float = 0.5  # seconds penalty on first lap of new tires
    warmup_decay_tau: float = 1.5  # laps for warmup penalty to decay exponentially
    warmup_model: Literal["exponential", "step"] = "exponential"
    warmup_step_laps: int = 2  # laps affected if using step model

    # Safety car settings (enhanced SC/VSC modeling)
    safety_car_prob: float = 0.3  # probability of SC in race
    vsc_prob: float = 0.2  # probability of VSC in race
    max_sc_events: int = 2  # maximum SC events per race
    max_vsc_events: int = 2  # maximum VSC events per race

    # SC impact
    sc_lap_time_reduction: float = 0.30  # 30% lap time reduction under SC
    sc_duration_mean: int = 5  # laps
    sc_duration_std: int = 2  # laps
    sc_pit_advantage: float = 15.0  # extra seconds saved if pit under SC

    # VSC impact (different from SC)
    vsc_lap_time_reduction: float = 0.40  # 40% lap time reduction under VSC
    vsc_duration_mean: int = 3  # laps
    vsc_duration_std: int = 1  # laps
    vsc_pit_advantage: float = 8.0  # smaller advantage than SC

    # Traffic modeling settings
    enable_traffic_model: bool = False  # disabled by default (simplified proxy)
    track_type: TrackType = "permanent"  # affects traffic penalty
    base_traffic_penalty: float = 0.5  # seconds per lap in traffic
    traffic_penalty_street: float = 1.2  # multiplier for street circuits
    traffic_penalty_permanent: float = 0.8  # multiplier for permanent circuits
    expected_rejoin_gap: float = 5.0  # seconds gap on rejoin (user configurable)

    # Validation settings
    validation_test_size: float = 0.2  # fraction of data held out for validation
    validation_min_laps: int = 10  # minimum laps needed to validate
    traffic_detection_threshold: float = 2.5  # std devs for traffic lap detection

    # Strategy analysis settings
    risk_percentiles: tuple[float, float] = (5.0, 95.0)  # percentiles for risk analysis
    sensitivity_parameters: list[str] = field(
        default_factory=lambda: ["pit_loss_mean", "safety_car_prob", "deg_rate"]
    )

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
