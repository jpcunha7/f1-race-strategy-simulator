# F1 Race Strategy Simulator: Methodology

**Author:** João Pedro Cunha

---

## Overview

This document describes the mathematical models, Monte Carlo simulation approach, and validation methodology used in the F1 Race Strategy Simulator.

## 1. Tire Degradation Modeling

### Data Collection
- **Source:** FastF1 race data (lap times + tire information)
- **Stint extraction:** Identify continuous periods with same tire compound
- **Filtering:** Remove pit laps, out-laps, safety car periods, statistical outliers

### Degradation Models

#### Linear Model
```
LapTime(age) = baseline + degradation_rate * age
```
- Simplest model
- Assumes constant degradation per lap
- Works well for many compounds/tracks

#### Quadratic Model
```
LapTime(age) = a + b * age + c * age²
```
- Captures accelerating degradation
- Better for compounds with cliff behavior
- Requires sufficient stint length

#### Piecewise Model
```
LapTime(age) = baseline + rate_early * age     if age < threshold
             = baseline + rate_late * age      if age >= threshold
```
- Models early vs late stint phases
- Captures warmup + degradation phases

### Model Fitting
- **Regression:** Robust Huber regression (reduces outlier impact)
- **Cross-validation:** K-fold CV on stint data to select best model
- **Per-compound:** Separate model for SOFT, MEDIUM, HARD
- **Uncertainty:** Residual standard deviation captured

### Example Output
```
SOFT:     baseline=89.2s, deg_rate=0.048s/lap, R²=0.91, σ=0.21s
MEDIUM:   baseline=89.8s, deg_rate=0.032s/lap, R²=0.88, σ=0.18s
HARD:     baseline=90.5s, deg_rate=0.021s/lap, R²=0.85, σ=0.16s
```

## 2. Race Validation

### Validation Protocol
1. **Historical race selection:** Choose year + event + driver
2. **Stint extraction:** Extract actual stints (laps + compounds)
3. **Fit degradation:** Train model on that race's data
4. **Predict lap times:** For each lap in each stint
5. **Compute errors:** MAE, RMSE per compound and overall
6. **Residual analysis:** Identify systematic errors, traffic laps

### Metrics
- **MAE (Mean Absolute Error):** Average |predicted - actual|
- **RMSE (Root Mean Square Error):** √(mean((predicted - actual)²))
- **R² (Coefficient of Determination):** Fraction of variance explained

### Traffic Detection
Outlier laps (residual > 2σ) flagged as potential traffic/incident laps.

## 3. Monte Carlo Simulation

### Simulation Loop
For each strategy (stint plan + compounds):
```
Repeat N times (typically 1000-5000):
  total_time = 0
  for each stint:
    for each lap in stint:
      # Sample lap time from degradation model + noise
      lap_time = predict_lap_time(stint_age, compound) + noise

      # Apply safety car if sampled
      if safety_car_active:
        lap_time *= sc_reduction_factor

      total_time += lap_time

    # Add pit stop time (if not last stint)
    if not last_stint:
      pit_time = sample_pit_loss() + pit_advantage_if_sc()
      total_time += pit_time

  record(total_time)
```

### Uncertainty Sources
1. **Degradation noise:** Normal(0, σ_degradation)
2. **Pit loss variation:** Normal(pit_loss_mean, pit_loss_std)
3. **Safety car events:** Bernoulli(sc_probability)
4. **Safety car timing:** Uniform over race laps
5. **Safety car duration:** Normal(4 laps, 1 lap)

### Output Statistics
- Mean race time
- Standard deviation
- 5th, 50th, 95th percentiles
- Probability distributions
- Strategy comparison metrics

## 4. Undercut and Overcut Estimation

### Undercut Mechanism
Driver A pits earlier than Driver B:
- **Gain source:** Fresh tires = faster laps while B is on old tires
- **Loss source:** Pit time, tire warmup penalty

### Undercut Gain Formula
```
undercut_gain = 0
for lap in [pit_lap_A .. pit_lap_B]:
  A_time = predict_lap(fresh_tire_age, new_compound) + warmup_penalty(lap - pit_lap_A)
  B_time = predict_lap(old_tire_age, old_compound)
  undercut_gain += (B_time - A_time)

undercut_gain -= pit_loss_A

if undercut_gain > track_position_buffer:
  undercut_successful = True
```

### Tire Warmup Penalty
```
warmup_penalty(laps_since_pit) = warmup_seconds * exp(-laps_since_pit / tau)
```
- **warmup_seconds:** Typical 0.5-1.0s
- **tau:** Time constant, typically 1.5-2.0 laps
- Alternative: Step function (first 2 laps +0.8s, then 0)

### Overcut
Driver B stays out longer:
- **Gain source:** Track position maintained, avoids traffic on out-lap
- **Loss source:** More degraded tires vs Driver A's fresh tires

## 5. Safety Car and VSC Modeling

### Safety Car (SC)
- **Lap time reduction:** 30-40% slower
- **Pit advantage:** Large (can pit with minimal time loss)
- **Probability:** User-configurable (typical 20-40% per race)
- **Duration:** Normal(4 laps, 1 lap)

### Virtual Safety Car (VSC)
- **Lap time reduction:** 35-45% slower
- **Pit advantage:** Small (still lose time but less than normal)
- **Probability:** User-configurable (typical 10-20% per race)
- **Duration:** Normal(3 laps, 1 lap)

### Implementation
```
if SC occurs at lap L:
  for lap in [L .. L+duration]:
    lap_time *= 0.65  # 35% reduction

  if driver_pits_during_SC:
    pit_advantage = normal_pit_loss * 0.3  # Save 70% of pit time
```

## 6. Traffic Modeling

### Simplified Approach
- **Input:** Expected rejoin gap after pit stop
- **Penalty:** If gap < threshold, apply traffic penalty
- **Penalty distribution:** Normal(mean=0.3s, std=0.1s) per lap in traffic

### Limitations
- Does NOT simulate full grid positions
- Does NOT model overtaking dynamics
- Simple proxy for portfolio demonstration

## 7. Strategy Optimization

### Grid Search
1. **Generate candidates:** All valid 1-stop and 2-stop strategies
2. **Constraints:**
   - Minimum stint length (default: 5 laps)
   - Maximum stint length (default: race_laps - 5)
   - Total laps = race distance
   - Optional: Must use 2 different compounds
3. **Evaluate:** Run Monte Carlo for each candidate
4. **Rank:** By mean race time

### Decision Metrics
- **Mean time:** Primary ranking
- **Risk:** 95th percentile - 5th percentile
- **Dominance:** P(Strategy A beats Strategy B)
- **Sensitivity:** Which parameter shifts ranking most

### Executive Summary
Top 3 strategies with:
- Mean ± 95% confidence interval
- Best-case / Worst-case scenarios
- Risk assessment
- Key decision drivers

## 8. Limitations and Assumptions

### What We Model
- Tire degradation from historical data
- Pit stop time variation
- Safety car probability and timing
- Basic tire warmup
- Simplified traffic penalties

### What We DON'T Model
- **Weather changes:** Rain, temperature variation
- **Full multi-driver simulation:** Only single driver focus
- **Incidents:** Crashes, red flags, mechanical failures
- **Fuel load effects:** Fuel weight impact on lap time
- **Track evolution:** Grip improvement over session
- **Detailed tire physics:** Graining, blistering, temperature
- **Driver skill variation:** Assumes consistent pace

### Interpretation
- Use for **comparative analysis** of strategies
- Predictions are **probabilistic ranges**, not exact times
- Real races have many factors not captured in model
- Validation against historical races shows typical accuracy ±2-5 seconds

## 9. Reproducibility

### Deterministic Execution
- All random sampling uses seeded RNG
- Seed recorded in run configuration
- Same seed = identical results

### Run Configuration
Every analysis saves:
- `run_config.json`: All parameters, seed, versions
- `summary.json`: Results and statistics
- `report.html`: Full visualization

### Outputs Structure
```
outputs/<run_id>/
  ├── run_config.json
  ├── summary.json
  ├── report.html
  └── figures/
      ├── degradation_curves.png
      ├── strategy_comparison.png
      └── undercut_heatmap.png
```

---

## References

1. **FastF1 Documentation:** https://docs.fastf1.dev/
2. **F1 Strategy Principles:** Pirelli tire data, FIA technical documents
3. **Monte Carlo Methods:** Standard simulation techniques

---

**Document Version:** 1.0
**Last Updated:** 2025-12-26
**Author:** João Pedro Cunha
