# F1 Race Strategy Simulator

**Degradation Modeling + Pit Stop Optimization with Monte Carlo Uncertainty**

A data science tool for simulating and optimizing Formula 1 race strategies using historical tire degradation data, Monte Carlo simulation, and physics-based reasoning.

**Author:** João Pedro Cunha
**License:** MIT

---

## Overview

This simulator helps answer questions like:
- *What's the optimal pit stop strategy for this race?*
- *Should I do a 1-stop or 2-stop strategy?*
- *What's the best pit window given tire degradation?*
- *How does safety car probability affect strategy choice?*

**Key Features:**
- 📊 **Tire Degradation Modeling**: Learn degradation curves from historical race stints
- 🎲 **Monte Carlo Simulation**: Account for uncertainty in degradation, pit loss, and safety cars
- ⚡ **Strategy Optimization**: Grid search over pit windows and compound combinations
- 📈 **Rich Visualizations**: Degradation curves, race time distributions, pit window heatmaps
- 🎯 **Physics-Based**: Models grip loss → lap time increase

**100% Free**: Uses only open-source tools and free FastF1 data.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry (recommended) or pip

### Using Poetry

```bash
git clone https://github.com/yourusername/f1-race-strategy-simulator.git
cd f1-race-strategy-simulator
poetry install
poetry shell
```

---

## Quick Start

### Streamlit Dashboard

```bash
poetry run streamlit run app/streamlit_app.py
```

Then open http://localhost:8501

**Usage:**
1. Select year, event (e.g., "Bahrain"), and driver code
2. Configure simulation settings (number of simulations, random seed)
3. Click "Run Analysis"
4. Explore results across multiple tabs
5. Download HTML report

### CLI Tool

```bash
# Basic usage
poetry run f1strategy run --year 2024 --event "Bahrain" --driver "VER" --n-sims 2000

# With custom seed
poetry run f1strategy run --year 2023 --event "Monaco" --driver "LEC" --n-sims 1000 --seed 123
```

**Output:** Creates a directory in `outputs/<run_id>/` containing:
- `summary.json`: Strategy comparison data
- `report.html`: Interactive HTML report with all visualizations

---

## Modeling Approach

### 1. Tire Degradation Model

**Physics Intuition:**
As tires wear, rubber temperature and grip decrease → lap times increase

**Model:**
```
LapTime(stint_age) = baseline + degradation_rate * stint_age
```

**Options:**
- **Linear**: `LapTime = a + b * age`
- **Quadratic**: `LapTime = a + b * age + c * age²`
- **Piecewise**: Different rates for early/late stint

**Fitting:**
- Uses robust regression (Huber) to handle outliers
- Automatically filters pit laps, out-laps, and statistical outliers
- Separate model per compound (SOFT, MEDIUM, HARD)
- Uncertainty captured via residual standard deviation

**Example Output:**
```
SOFT: baseline=89.2s, deg_rate=0.048s/lap, R²=0.91
MEDIUM: baseline=89.8s, deg_rate=0.032s/lap, R²=0.88
```

### 2. Race Simulation

Each simulation:
1. **Lap-by-lap execution**: For each stint, predict lap time using degradation model + noise
2. **Pit stops**: Add pit loss (sampled from normal distribution)
3. **Safety car**: Probabilistic SC events reduce lap times + provide pit advantage
4. **Total time**: Sum all lap times + pit losses

**Uncertainty Sources:**
- Degradation variation (around fitted curve)
- Pit loss variation (typical: 22±1.5s)
- Safety car occurrence and timing
- Safety car duration

### 3. Strategy Optimization

**Grid Search:**
- Generate all valid 1-stop and 2-stop strategies
- For each: run Monte Carlo simulation (typically 1000-5000 iterations)
- Rank by mean race time
- Return top strategies with confidence intervals

**Constraints:**
- Minimum stint length (configurable, default: 5 laps)
- Total laps must equal race distance
- Optional: require two different compounds

---

## Modeling Assumptions & Limitations

### What We Model
✅ Tire degradation from historical data
✅ Pit stop time variation
✅ Safety car probability and timing
✅ Uncertainty quantification via Monte Carlo

### What We DON'T Model
❌ **Traffic**: Assumes clear air throughout
❌ **Weather changes**: Rain/wet conditions
❌ **Incidents**: Crashes, VSC, red flags (beyond SC)
❌ **Driver skill variation**: Assumes consistent performance
❌ **Fuel load**: Fuel effect on lap time
❌ **Track evolution**: Grip changes over session
❌ **Tire temperature**: Warm-up laps, graining, blistering

### Interpretation Guidelines

✅ **Do:** Use for comparative strategy analysis
✅ **Do:** Understand relative trade-offs (1-stop vs 2-stop)
✅ **Do:** Identify optimal pit windows

❌ **Don't:** Treat predictions as absolute race times
❌ **Don't:** Ignore real-world factors (traffic, weather)
❌ **Don't:** Use for betting or commercial purposes

---

## Project Structure

```
f1-race-strategy-simulator/
├── src/f1strategy/
│   ├── config.py          # Configuration and settings
│   ├── data_loader.py     # FastF1 data loading, stint extraction
│   ├── degrade_model.py   # Tire degradation fitting
│   ├── simulator.py       # Monte Carlo race simulation
│   ├── optimizer.py       # Strategy optimization (grid search)
│   ├── viz.py             # Plotly visualizations
│   ├── report.py          # HTML report generation
│   └── cli.py             # Command-line interface
├── app/
│   └── streamlit_app.py   # Interactive Streamlit dashboard
├── tests/
│   ├── test_degrade_model.py
│   └── test_simulator.py
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Examples

### Example 1: Bahrain 2024 - Verstappen

```bash
poetry run f1strategy run --year 2024 --event "Bahrain" --driver "VER" --n-sims 2000
```

**Output:** Recommends 1-stop SOFT→HARD strategy, pitting on lap 18-22

### Example 2: Monaco 2023 - Interactive Analysis

```bash
poetry run streamlit run app/streamlit_app.py
```

Select Monaco, explore degradation curves for SOFT/MEDIUM, compare 1-stop vs 2-stop strategies with uncertainty bands.

---

## Visualizations

The tool generates:

1. **Degradation Curves**: Lap time vs stint age with confidence bands
2. **Race Time Distributions**: Histogram of simulated race times per strategy
3. **Cumulative Distribution**: CDF showing probability of finishing under time X
4. **Strategy Comparison**: Bar chart with mean times and error bars
5. **Pit Window Heatmap**: Optimal pit lap for given compound combination
6. **Sensitivity Analysis**: Impact of parameter changes (pit loss, degradation, SC probability)

---

## Development

### Running Tests

```bash
poetry run pytest
poetry run pytest --cov=f1strategy --cov-report=term-missing
```

### Code Quality

```bash
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/f1strategy --ignore-missing-imports
```

### CI/CD

GitHub Actions runs tests on Python 3.10, 3.11, 3.12 for every push/PR.

---

## Data Source

**FastF1** (https://docs.fastf1.dev/)
- Free, open-source Python package
- Provides F1 timing, telemetry, and race data
- Supports 2018-2025 seasons
- Data cached locally to avoid re-downloads

---

## Troubleshooting

### "No degradation models could be fitted"
- Some races have insufficient clean stint data
- Try a different event with more typical tire strategy
- Check driver actually completed significant race distance

### "Session not found"
- Verify event name matches exactly (e.g., "Bahrain" not "Sakhir")
- Try using round number instead of name
- Ensure race session ("R") exists for that event

---

## Roadmap

Future enhancements:
- [ ] Weather/rain modeling
- [ ] Traffic impact simulation
- [ ] Multi-driver race simulation
- [ ] Lap-by-lap position tracking
- [ ] Real-time strategy updates during race
- [ ] ML-based degradation prediction

---

## Acknowledgments

- **FastF1**: Free F1 data access
- **Plotly & Streamlit**: Interactive visualizations and dashboards
- **F1 Strategy Community**: Inspiration and domain knowledge

---

## License

MIT License - see LICENSE for details

Copyright (c) 2025 João Pedro Cunha

---

## Citation

If you use this tool in research or presentations:

```
Cunha, J.P. (2025). F1 Race Strategy Simulator: Degradation Modeling +
Pit Stop Optimization. https://github.com/yourusername/f1-race-strategy-simulator
```

---

**Built with Claude Code** | Data powered by FastF1
