# Real-Time Edge Vision & Analytics Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-teal.svg)](https://seaborn.pydata.org/)

An edge computing analytics pipeline designed to ingest simulated live video streams, isolate spatial motion vectors via OpenCV and NumPy, aggregate high-throughput tracking telemetry using Pandas, and synthesize behavioral dashboards using Seaborn.

---

## Output Dashboard Preview

![Analytics Dashboard](assets/analytics_dashboard.png)

---

## Key Highlights

- **Real-Time Edge Processing:** Runs background subtraction (`MOG2`) and morphological kernel transformations at 30+ FPS equivalent throughput.
- **Vectorized Spatial Tracking:** Leverages NumPy arrays for low-latency centroid estimation, velocity calculation, and frame-boundary collision tracking.
- **Telemetry Aggregation:** Ingests frame-by-frame coordinate detections into Pandas DataFrames, aggregating dwell times, tracking densities, and zone retention scores.
- **Statistical Visualization:** Generates a 4-panel report displaying 2D spatial occupancy density (KDE), bounding box log-variance, and temporal rolling interaction windows.

---

## Architecture Overview
