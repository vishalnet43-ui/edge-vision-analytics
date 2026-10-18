"""
core/vision_engine.py
Handles frame acquisition, background subtraction, and spatial vector telemetry.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class DetectionEvent:
    frame_id: int
    timestamp_ms: float
    track_id: int
    center_x: float
    center_y: float
    bounding_area: float
    zone: str


class VisionPipeline:
    def __init__(self, frame_dims: Tuple[int, int] = (640, 480)):
        self.width, self.height = frame_dims
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def _determine_zone(self, x: float, y: float) -> str:
        """Vectorized spatial zone assignment based on frame dimensions."""
        if x < self.width / 2:
            return "Zone_A_Display" if y < self.height / 2 else "Zone_C_Checkout"
        return "Zone_B_Aisle" if y < self.height / 2 else "Zone_D_Entrance"

    def process_frame(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> Tuple[np.ndarray, List[DetectionEvent]]:
        """Processes a frame, applies background subtraction, and extracts spatial metrics."""
        # Convert to grayscale and apply Gaussian blur to minimize sensor noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)

        # Morphological opening and dilation to merge close contours
        cleaned_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        dilated_mask = cv2.dilate(cleaned_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[DetectionEvent] = []
        track_counter = 0

        annotated_frame = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 600:  # Noise threshold filter
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
            zone = self._determine_zone(cx, cy)

            detections.append(
                DetectionEvent(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    track_id=track_counter,
                    center_x=cx,
                    center_y=cy,
                    bounding_area=float(area),
                    zone=zone,
                )
            )
            track_counter += 1

            # Render bounding box & tracking indicators on output frame
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 127), 2)
            cv2.circle(annotated_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                annotated_frame,
                f"{zone}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        return annotated_frame, detections