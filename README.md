# F1 Race Strategy Simulator

**Professional Strategy Analysis Tool for Formula 1**

A comprehensive Monte Carlo simulation system for F1 race strategy optimization, featuring physics-based tire degradation modeling, undercut/overcut analysis, and rigorous model validation.

**Author:** João Pedro Cunha

---

## Overview

This tool provides race engineers and strategists with data-driven insights for optimal pit strategy decisions. Built on statistical modeling and probabilistic simulation, it accounts for tire degradation, safety car events, pit stop timing, and strategic interactions.

### Key Features

**Core Analysis:**
- Automatic degradation model selection (linear/quadratic/piecewise)
- Monte Carlo simulation with uncertainty quantification
- Strategy optimization with risk assessment
- Model validation against historical performance

**Strategic Tools:**
- Undercut/overcut window analysis with warmup penalties
- Safety Car and Virtual Safety Car modeling
- Traffic penalty estimation
- Sensitivity analysis for decision robustness

**Professional Interface:**
- Multi-page Streamlit dashboard
- Command-line interface with validation and undercut commands
- Executive summaries and risk profiles
- Publication-quality visualizations

---

## Installation

```bash
# Clone repository
git clone https://github.com/jpcunha7/f1-race-strategy-simulator.git
cd f1-race-strategy-simulator

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python 3.9+
- fastf1
- numpy, pandas, scipy, scikit-learn
- plotly
- streamlit
- tqdm

---

## Usage

### Streamlit Dashboard

Launch the professional multi-page interface:

```bash
streamlit run app/streamlit_app.py
```

**Pages:**
1. **Overview** - Executive summary and top strategy recommendations
2. **Degradation Models** - Tire degradation curves with model parameters
3. **Model Validation** - Prediction accuracy and credibility metrics
4. **Strategy Comparison** - Performance comparison with risk profiles
5. **Pit Window Explorer** - Interactive undercut/overcut analysis
6. **Scenario Analysis** - Sensitivity testing (planned)

### Command-Line Interface

**Full Strategy Optimization:**
```bash
f1strategy optimize --year 2024 --event "Bahrain" --driver "VER" --n-sims 2000
```

**Model Validation:**
```bash
f1strategy validate --year 2024 --event "Bahrain" --driver "VER"
```

**Undercut Analysis:**
```bash
f1strategy undercut --year 2024 --event "Bahrain" --driver "VER" --lap 15
```

**Quick Mode:**
```bash
f1strategy optimize --year 2024 --event "Monaco" --driver "LEC" --quick
```

---

## Technical Architecture

### 1. Tire Degradation Modeling

**Physics Reasoning:**
Tire degradation causes lap time increase due to progressive grip loss. Different compounds and track conditions produce distinct degradation patterns.

**Model Types:**
- **Linear:** Constant degradation rate (most common)
  - `LapTime(age) = baseline + rate × age`
- **Quadratic:** Accelerating degradation (tire cliff)
  - `LapTime(age) = baseline + rate × age + a × age²`
- **Piecewise:** Different rates in early vs late stint
  - Separate linear models before/after breakpoint

**Selection:**
Automatic cross-validated model selection chooses the best fit for each compound.

**Implementation:** `/src/f1strategy/degrade_model.py`

### 2. Race Simulation

Monte Carlo approach samples from degradation uncertainty, pit stop variability, and safety car probabilities to generate race time distributions.

**Key Features:**
- Tire warmup penalties (exponential or step model)
- SC/VSC events with distinct characteristics
- Pit loss with normal distribution
- Deterministic with seeds for reproducibility

**Implementation:** `/src/f1strategy/simulator.py`

### 3. Model Validation

**Critical for Credibility:**
Train/test split on historical race data validates prediction accuracy.

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² by compound
- Traffic lap detection

**Implementation:** `/src/f1strategy/validation.py`

### 4. Undercut/Overcut Analysis

**Strategy Reasoning:**
Undercut works when fresh tire pace exceeds degradation of opponent's worn tires, accounting for pit loss and warmup penalty.

**Calculation:**
```
Net Gain = Σ(opponent_laptime - your_laptime) - pit_loss - warmup_penalty
```

**Implementation:** `/src/f1strategy/undercut.py`

### 5. Risk Analysis

**Strategy Dominance:**
Probability that Strategy A beats Strategy B across all scenarios.

**Risk Profiles:**
- Best case (5th percentile)
- Mean expected time
- Worst case (95th percentile)
- Probability of being fastest

**Implementation:** `/src/f1strategy/optimizer.py`

---

## Configuration

All parameters configurable via `StrategyConfig`:

```python
from f1strategy.config import StrategyConfig

config = StrategyConfig(
    # Monte Carlo
    n_simulations=2000,
    random_seed=42,

    # Degradation
    auto_select_degradation_model=True,
    cv_folds=3,

    # Pit stops
    pit_loss_mean=22.0,
    pit_loss_std=1.5,

    # Warmup
    warmup_penalty_initial=0.5,
    warmup_decay_tau=1.5,
    warmup_model="exponential",

    # Safety cars
    safety_car_prob=0.3,
    vsc_prob=0.2,
    max_sc_events=2,
    max_vsc_events=2,

    # SC/VSC impact
    sc_lap_time_reduction=0.30,
    vsc_lap_time_reduction=0.40,
    sc_pit_advantage=15.0,
    vsc_pit_advantage=8.0,

    # Traffic
    enable_traffic_model=False,
    track_type="permanent",

    # Validation
    validation_test_size=0.2,
    traffic_detection_threshold=2.5,
)
```

---

## Module Reference

### Core Modules

**`config.py`** - Configuration dataclass with all parameters

**`data_loader.py`** - FastF1 integration for race data extraction

**`degrade_model.py`** - Tire degradation modeling with automatic selection

**`simulator.py`** - Monte Carlo race simulation with SC/VSC

**`optimizer.py`** - Strategy generation, optimization, and risk analysis

### Analysis Modules

**`validation.py`** - Model validation against historical performance

**`undercut.py`** - Undercut/overcut gain estimation and pit window analysis

**`traffic.py`** - Traffic penalty modeling (simplified proxy)

### Visualization & Interface

**`viz.py`** - Professional F1-themed Plotly visualizations

**`cli.py`** - Command-line interface

**`app/streamlit_app.py`** - Multi-page Streamlit dashboard

**`report.py`** - HTML report generation

---

## Example Workflow

```python
from f1strategy import (
    config,
    data_loader,
    degrade_model,
    optimizer,
    validation,
    undercut,
)

# 1. Load race data
cfg = config.StrategyConfig(n_simulations=2000, random_seed=42)
session = data_loader.load_race_session(2024, "Bahrain", cfg)
race_info = data_loader.get_race_info(session)
stint_data = data_loader.extract_stints(session, "VER")

# 2. Fit and validate degradation models
deg_models = degrade_model.fit_all_compounds(stint_data, cfg)

validation_result = validation.validate_race(
    2024, "Bahrain", "VER", stint_data, cfg
)
print(f"Validation MAE: {validation_result.overall_mae:.3f}s")

# 3. Optimize strategies
ranked_strategies, results_dict = optimizer.optimize_strategy(
    deg_models,
    race_info['total_laps'],
    cfg,
)

# 4. Calculate risk profiles
risk_profiles = optimizer.calculate_risk_profiles(results_dict, cfg)

# 5. Generate executive summary
summary = optimizer.create_strategy_executive_summary(
    results_dict,
    risk_profiles,
    top_n=3,
)
print(summary)

# 6. Analyze undercut windows
compounds = list(deg_models.keys())
recommendation = undercut.find_optimal_undercut_window(
    race_laps=race_info['total_laps'],
    your_compound=compounds[0],
    your_stint_age_start=10,
    new_compound=compounds[1],
    opponent_compound=compounds[0],
    opponent_stint_age_start=10,
    degradation_models=deg_models,
    config=cfg,
)

print(f"Optimal undercut: Lap {recommendation.optimal_lap}")
print(f"Expected gain: {recommendation.expected_gain:+.2f}s")
```

---

## Design Philosophy

### Professional Tool for Race Engineers

This simulator is designed to look and function like actual strategy tools used in F1 paddocks:

1. **Physics-Based**: All models grounded in tire physics and race dynamics
2. **Validated**: Credibility through rigorous testing on historical data
3. **Probabilistic**: Acknowledges uncertainty via Monte Carlo methods
4. **Decision-Focused**: Executive summaries, risk profiles, sensitivity analysis
5. **Professional UI**: Clean, dark theme, F1 red accents, no emojis

### No Emojis Policy

Professional tools for engineers use clear, technical language without decorative elements. All user-facing text is direct and informative.

---

## Data Source

Race data provided by [FastF1](https://github.com/theOehrly/Fast-F1), an open-source Python library for accessing F1 timing data.

---

## Future Enhancements

- Full grid position simulation for traffic modeling
- Real-time race strategy updates during live sessions
- Multi-car strategy coordination
- Weather impact modeling
- Fuel load effects on lap times
- Integration with telemetry data

---

## License

MIT License - See LICENSE file

---

## Author

**João Pedro Cunha**

Professional F1 Race Strategy Analysis Tool

For questions or contributions, please open an issue or pull request.

---

## Acknowledgments

- FastF1 project for F1 data access
- F1 strategy community for domain expertise
- Open-source scientific Python ecosystem
