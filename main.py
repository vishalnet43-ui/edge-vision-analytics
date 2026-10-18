"""
main.py
Main driver script demonstrating end-to-end edge analytics execution.
"""

import time
from core.vision_engine import VisionPipeline
from analytics.processor import AnalyticsProcessor
from analytics.visualizer import AnalyticsVisualizer
import numpy as np


def generate_synthetic_stream(num_frames: int = 180, width: int = 640, height: int = 480):
    """
    Simulates live video by rendering dynamic objects with 
    Gaussian noise and linear kinematic trajectories.
    """
    fps = 30.0
    # Create persistent synthetic targets with initial coordinates and velocities
    targets = [
        {"x": 100.0, "y": 80.0, "vx": 2.5, "vy": 1.8, "size": 35},
        {"x": 500.0, "y": 400.0, "vx": -3.0, "vy": -1.2, "size": 45},
        {"x": 320.0, "y": 150.0, "vx": 1.2, "vy": 2.2, "size": 30},
    ]

    for frame_id in range(num_frames):
        # Base ambient frame with noise
        frame = np.random.normal(40, 5, (height, width, 3)).astype(np.uint8)

        # Update synthetic kinematic positions
        for target in targets:
            target["x"] += target["vx"] + np.random.normal(0, 0.5)
            target["y"] += target["vy"] + np.random.normal(0, 0.5)

            # Reflect off frame boundaries
            if target["x"] <= 30 or target["x"] >= width - 30:
                target["vx"] *= -1
            if target["y"] <= 30 or target["y"] >= height - 30:
                target["vy"] *= -1

            cx, cy = int(target["x"]), int(target["y"])
            s = target["size"]
            # Render synthetic object
            frame[
                max(0, cy - s) : min(height, cy + s),
                max(0, cx - s) : min(width, cx + s)
            ] = 220

        timestamp_ms = (frame_id / fps) * 1000.0
        yield frame, frame_id, timestamp_ms


def main():
    print("=" * 70)
    print("Starting Edge Vision & Spatial Analytics Pipeline")
    print("=" * 70)

    # 1. Initialize Modules
    vision = VisionPipeline(frame_dims=(640, 480))
    processor = AnalyticsProcessor(fps=30.0)
    visualizer = AnalyticsVisualizer(output_dir="data")

    all_detections = []
    start_wall_time = time.time()

    # 2. Ingest Stream & Run Feature Extraction
    print("[1/3] Ingesting video frames and tracking bounding geometries...")
    for frame, frame_id, timestamp_ms in generate_synthetic_stream(num_frames=240):
        _, detections = vision.process_frame(frame, frame_id, timestamp_ms)
        all_detections.extend(detections)

    elapsed = time.time() - start_wall_time
    fps = 240 / elapsed
    print(f"Processed 240 frames in {elapsed:.2f}s ({fps:.1f} FPS equivalent)")

    # 3. Pandas Transformation & Aggregation
    print("[2/3] Transforming spatial vectors & computing Zone KPIs...")
    raw_df = processor.build_dataframe(all_detections)
    kpi_df = processor.compute_zone_kpis(raw_df)

    print("\n--- Operational Zone KPI Summary Table ---")
    print(kpi_df.to_string(index=False))

    # 4. Generate Seaborn Dashboard
    print("\n[3/3] Generating statistical distribution dashboard...")
    saved_chart = visualizer.generate_dashboard(raw_df, kpi_df)
    print(f"Dashboard saved successfully to: {saved_chart.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()