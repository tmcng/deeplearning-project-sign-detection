"""
Visualization module for rendering detections, tracking IDs, ROI scan zones, and telemetry HUD.
"""
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np


class StreamVisualizer:
    """Renders professional overlays onto video frames."""

    def __init__(self, box_color: Tuple[int, int, int] = (0, 220, 0),
                 roi_color: Tuple[int, int, int] = (255, 180, 0),
                 hud_color: Tuple[int, int, int] = (0, 255, 255)):
        self.box_color = box_color
        self.roi_color = roi_color
        self.hud_color = hud_color

    def draw_roi(self, frame: np.ndarray, roi_pixels: Tuple[int, int, int, int]) -> None:
        """Draw scanning area boundaries for domain-specific crop."""
        rx1, ry1, rx2, ry2 = roi_pixels
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), self.roi_color, 1)
        cv2.putText(frame, "ROI scan area", (rx1 + 5, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.roi_color, 1, cv2.LINE_AA)

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]],
                        has_tracking: bool = True) -> None:
        """Draw bounding boxes, class labels, confidence scores, and tracking IDs."""
        for d in detections:
            x1, y1, x2, y2 = map(round, d['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

            label = f"class {d['class_id']} {d['score']:.2f}"
            if has_tracking and 'track_id' in d and d['track_id'] is not None:
                label += f" ID {d['track_id']}"

            # Label banner
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_y = max(text_h + 5, y1 - 4)
            cv2.rectangle(frame, (x1, text_y - text_h - 2), (x1 + text_w + 2, text_y + baseline - 1),
                          (20, 20, 20), cv2.FILLED)
            cv2.putText(frame, label, (x1 + 1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.box_color, 1, cv2.LINE_AA)

    def draw_hud(self, frame: np.ndarray, model_name: str, mode_desc: str,
                 infer_fps: float, frame_id: int) -> None:
        """Render top-left telemetry heads-up display (HUD)."""
        banner_text = f"{model_name} ({mode_desc}) | {infer_fps:.2f} FPS | Frame #{frame_id}"
        # Dark backdrop for legibility
        (text_w, text_h), baseline = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (10, 8), (20 + text_w, 15 + text_h + baseline), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, banner_text, (15, 10 + text_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hud_color, 2, cv2.LINE_AA)
