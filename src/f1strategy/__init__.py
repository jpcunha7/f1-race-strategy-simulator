"""
F1 Race Strategy Simulator: Degradation Modeling + Pit Stop Optimization

A comprehensive strategy simulation engine for Formula 1 races with:
- Tire degradation modeling from historical data
- Monte Carlo simulation with uncertainty
- Pit stop optimization
- Physics-based explanations

Author: João Pedro Cunha
License: MIT
"""

__version__ = "0.1.0"
__author__ = "João Pedro Cunha"

from f1strategy import config, data_loader, degrade_model, simulator, optimizer, viz

__all__ = [
    "config",
    "data_loader",
    "degrade_model",
    "simulator",
    "optimizer",
    "viz",
]
