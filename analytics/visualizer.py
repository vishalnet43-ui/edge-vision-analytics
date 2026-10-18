"""
analytics/visualizer.py
Seaborn and Matplotlib visualization suite for spatial interaction analytics.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class AnalyticsVisualizer:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="darkgrid", context="paper", font_scale=1.1)

    def generate_dashboard(
        self, raw_df: pd.DataFrame, kpi_df: pd.DataFrame, filename: str = "analytics_dashboard.png"
    ) -> Path:
        """Produces a 4-panel statistical dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        palette = sns.color_palette("mako", as_cmap=False)

        # 1. Spatial Kernel Density Estimate (Spatial Interaction Heatmap)
        sns.kdeplot(
            data=raw_df,
            x="center_x",
            y="center_y",
            fill=True,
            cmap="viridis",
            thresh=0.05,
            levels=15,
            alpha=0.85,
            ax=axes[0, 0],
        )
        axes[0, 0].set_xlim(0, 640)
        axes[0, 0].set_ylim(480, 0)  # Invert to match OpenCV coordinate space
        axes[0, 0].set_title("Spatial Occupancy Density (2D KDE)", fontweight="bold")
        axes[0, 0].set_xlabel("X-Coordinate (px)")
        axes[0, 0].set_ylabel("Y-Coordinate (px)")

        # 2. Zone Dwell & Detection Count
        sns.barplot(
            data=kpi_df,
            x="zone",
            y="total_detections",
            hue="zone",
            legend=False,
            palette="crest",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Total Tracked Events per Operational Zone", fontweight="bold")
        axes[0, 1].set_ylabel("Detection Frequency")
        axes[0, 1].set_xlabel("Zone")

        # 3. Object Scale / Area Distribution (Boxen Plot with Log Scale)
        sns.boxenplot(
            data=raw_df,
            x="zone",
            y="bounding_area",
            hue="zone",
            legend=False,
            palette="rocket",
            ax=axes[1, 0],
        )
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title("Bounding Box Area Variance (Log Scale)", fontweight="bold")
        axes[1, 0].set_ylabel("Area (px²)")
        axes[1, 0].set_xlabel("Zone")

        # 4. Temporal Interaction Intensity (Rolling Moving Average)
        time_binned = (
            raw_df.set_index(pd.to_datetime(raw_df["time_sec"], unit="s"))
            .groupby([pd.Grouper(freq="2s"), "zone"])
            .size()
            .unstack(fill_value=0)
        )

        time_binned.plot(ax=axes[1, 1], lw=2.2, alpha=0.9)
        axes[1, 1].set_title("Temporal Event Density (2-Sec Windows)", fontweight="bold")
        axes[1, 1].set_xlabel("Relative Timeline")
        axes[1, 1].set_ylabel("Events / Sec")
        axes[1, 1].legend(title="Zone", loc="upper right")

        plt.tight_layout()
        export_path = self.output_dir / filename
        plt.savefig(export_path, dpi=300)
        plt.close(fig)

        return export_path