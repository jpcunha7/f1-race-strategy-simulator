"""Data loading module for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import logging
from typing import Union

import fastf1
import pandas as pd
from fastf1.core import Session

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig

logger = logging.getLogger(__name__)


def enable_cache(cache_dir: str = "cache") -> None:
    """Enable FastF1 caching."""
    try:
        fastf1.Cache.enable_cache(cache_dir)
        logger.info(f"FastF1 cache enabled at: {cache_dir}")
    except Exception as e:
        logger.warning(f"Failed to enable cache: {e}")


def load_race_session(
    year: int,
    event: Union[int, str],
    config: StrategyConfig = DEFAULT_CONFIG,
) -> Session:
    """Load F1 race session."""
    try:
        if config.enable_cache:
            enable_cache(str(config.cache_dir))

        logger.info(f"Loading race session: {year} {event}")
        session = fastf1.get_session(year, event, "R")
        session.load()
        logger.info(f"Race session loaded: {session.event['EventName']}")
        return session

    except Exception as e:
        error_msg = f"Failed to load race session {year} {event}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def extract_stints(session: Session, driver: str) -> pd.DataFrame:
    """Extract stint information for a driver.

    Returns DataFrame with columns:
    - Stint: stint number
    - Compound: tire compound
    - LapNumber: lap number
    - LapTime: lap time in seconds
    - StintAge: age of stint (laps on this tire)
    - IsOutlier: flagged if appears to be traffic/issue
    """
    try:
        laps = session.laps.pick_drivers(driver).copy()

        if laps.empty:
            raise ValueError(f"No laps found for driver {driver}")

        # Add stint information
        laps = laps.reset_index(drop=True)

        # Create stint age based on compound changes
        laps["Compound"] = laps["Compound"].fillna("UNKNOWN")
        laps["StintChange"] = (laps["Compound"] != laps["Compound"].shift(1)).astype(int)
        laps["Stint"] = laps["StintChange"].cumsum()

        # Calculate stint age
        laps["StintAge"] = laps.groupby("Stint").cumcount() + 1

        # Convert lap time to seconds
        laps["LapTime"] = laps["LapTime"].dt.total_seconds()

        # Flag potential outliers (in-lap, out-lap, traffic)
        laps["IsOutlier"] = False

        # Mark pit laps
        laps.loc[laps["PitInTime"].notna(), "IsOutlier"] = True
        laps.loc[laps["PitOutTime"].notna(), "IsOutlier"] = True

        # Select relevant columns
        stint_data = laps[["Stint", "Compound", "LapNumber", "LapTime", "StintAge", "IsOutlier"]]

        logger.info(f"Extracted {len(stint_data)} laps across {laps['Stint'].nunique()} stints")

        return stint_data

    except Exception as e:
        error_msg = f"Failed to extract stints for {driver}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_race_info(session: Session) -> dict:
    """Extract race information."""
    total_laps = int(session.total_laps) if hasattr(session, "total_laps") else 50

    return {
        "event_name": session.event.get("EventName", "Unknown"),
        "location": session.event.get("Location", "Unknown"),
        "country": session.event.get("Country", "Unknown"),
        "date": str(session.date) if session.date else "Unknown",
        "total_laps": total_laps,
    }
