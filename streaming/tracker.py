"""
Greedy, class-aware bounding box IoU tracker.
Designed for real-time video inference without heavy appearance descriptor dependencies.
"""
from typing import Dict, List, Tuple, Any


class IoUTracker:
    """Greedy, class-aware one-to-one track association.
    
    Tracks expire if not associated within max_lost_frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, Tuple[Dict[str, Any], int]] = {}
        self.next_id = 1

    @staticmethod
    def iou(a: List[float], b: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
        inter_x1 = max(a[0], b[0])
        inter_y1 = max(a[1], b[1])
        inter_x2 = min(a[2], b[2])
        inter_y2 = min(a[3], b[3])

        area_inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        area_union = area_a + area_b - area_inter

        return area_inter / area_union if area_union > 0 else 0.0

    def update(self, detections: List[Dict[str, Any]], frame_id: int) -> None:
        """Associate detections to active tracks or initialize new tracks in-place."""
        # Purge stale tracks exceeding max_lost_frames
        self.tracks = {
            track_id: (det, last_seen)
            for track_id, (det, last_seen) in self.tracks.items()
            if frame_id - last_seen <= self.max_lost_frames
        }

        available_track_ids = set(self.tracks.keys())

        # Sort detections descending by confidence score
        for d in sorted(detections, key=lambda x: x['score'], reverse=True):
            candidates = [
                (self.iou(d['box'], self.tracks[t_id][0]['box']), t_id)
                for t_id in available_track_ids
                if self.tracks[t_id][0]['class_id'] == d['class_id']
            ]

            best_iou, matched_id = max(candidates, default=(0.0, -1))
            if best_iou < self.iou_threshold or matched_id == -1:
                # Assign a fresh ID
                assigned_id = self.next_id
                self.next_id += 1
            else:
                assigned_id = matched_id
                available_track_ids.remove(matched_id)

            d['track_id'] = assigned_id
            self.tracks[assigned_id] = (d.copy(), frame_id)
