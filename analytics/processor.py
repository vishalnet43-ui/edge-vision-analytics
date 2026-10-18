"""
analytics/processor.py
Pandas-powered data aggregation, feature engineering, and KPI computation.
"""

from typing import List
import numpy as np
import pandas as pd
from core.vision_engine import DetectionEvent


class AnalyticsProcessor:
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def build_dataframe(self, events: List[DetectionEvent]) -> pd.DataFrame:
        if not events:
            return pd.DataFrame()

        df = pd.DataFrame([e.__dict__ for e in events])
        # Calculate continuous time vectors
        df["time_sec"] = df["timestamp_ms"] / 1000.0
        return df

    def compute_zone_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates dwell duration, interaction count, and average spatial footprint."""
        if df.empty:
            return pd.DataFrame()

        # Grouping by Zone to extract statistical metrics
        grouped = df.groupby("zone")

        kpi_df = grouped.agg(
            total_detections=("frame_id", "count"),
            unique_tracks=("track_id", "nunique"),
            avg_area_px=("bounding_area", "mean"),
            std_area_px=("bounding_area", "std"),
            dwell_seconds=("time_sec", lambda x: (x.max() - x.min()) if len(x) > 1 else 0.0),
        ).reset_index()

        # Handle null values in standard deviations for single-element zones
        kpi_df["std_area_px"] = kpi_df["std_area_px"].fillna(0.0)

        # Compute engagement index: normalized detection density
        kpi_df["engagement_score"] = np.round(
            (kpi_df["total_detections"] / (kpi_df["unique_tracks"] + 1e-5))
            * np.log1p(kpi_df["avg_area_px"]),
            2,
        )

        return kpi_df.sort_values(by="engagement_score", ascending=False)