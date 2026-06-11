from typing import List, Tuple

import cv2
from cv2.typing import MatLike
import logfire
import numpy as np

from src.entities.models.app.video_item import VideoItem

class VideoPreprocessor:
    def __init__(self, match_id: int, n_samples: int = 5, histogram_threshold: float = 0.09, compression_width: int = 560):
        self.prev_gray = None
        self.prev_hist = None
        self.windows_frames: List[Tuple[int, MatLike, float, float]] = []
        self.match_id = match_id
        
        self.fps: float = 30
        self.n_samples = max(1, n_samples)
        
        self.histogram_threshold = histogram_threshold
        self.compression_width = compression_width
        self.scene_threshold = 0.09
        self.motion_threshold = 0.25
        self.max_skip_frames = 5
        self.frames_since_scene_change = 0

        self.resize_w = 556
        self.resize_h = 370
        self.resized = False

    def compress_frame(self, frame: MatLike) -> MatLike:
        if self.resized:
            return cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]
        if w <= self.compression_width:
            return frame
        
        new_h = int(h * self.compression_width / w)
        self.resize_h = new_h
        self.resize_w = self.compression_width
        self.resized = True

        return cv2.resize(frame, (self.compression_width, new_h), interpolation=cv2.INTER_AREA)

    def histogram_delta(self, frame: MatLike) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if self.prev_hist is None:
            self.prev_hist = hist
            return 0.0

        dist = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        self.prev_hist = hist
        logfire.debug(f"[VideoProcessor] Histogram distance: {dist:.4f}")
        return dist

    def add_frame(self, frame: MatLike, timestamp: float, frame_num: int) -> List[VideoItem]:
        small = self.compress_frame(frame)
        delta = self.histogram_delta(small)

        self.windows_frames.append((frame_num, frame, delta, timestamp))

        if len(self.windows_frames) >= self.fps:
            selected = self.select_samples_from_window()
            self.windows_frames.clear()
            return selected
        
        return []
    
    def flush(self):
        selected = self.select_samples_from_window()
        self.windows_frames.clear()
        return selected
    
    def select_samples_from_window(self) -> List[VideoItem]:
        
        if not self.windows_frames:
            return []
        
        if len(self.windows_frames) <= self.n_samples:
            return [VideoItem(
                annotated_frame=frame.copy(),
                frame=frame.copy(),
                frame_num=idx,
                timestamp=dt,
                match_id=self.match_id
            ) for idx, frame, _, dt in self.windows_frames]
        
        stable_frames = [
            item for item in self.windows_frames
            if item[2] <= self.histogram_threshold
        ]

        candidates = stable_frames if stable_frames else self.windows_frames
        candidates = sorted(candidates, key=lambda item: item[2])

        min_gap = max(1, self.fps // self.n_samples)
        selected: List[VideoItem] = []
        selected_indices = []

        for idx, frame, delta, timestamp in candidates:
            if len(selected) >= self.n_samples:
                break

            if any(abs(idx - prev_idx) < min_gap for prev_idx in selected_indices):
                continue
            
            video_item = VideoItem(
                annotated_frame=frame.copy(),
                frame=frame.copy(),
                frame_num=idx,
                timestamp=timestamp,
                match_id=self.match_id
            )
            selected.append(video_item)
            selected_indices.append(idx)
        
        logfire.debug(f"[VideoProcessor] Selected {len(selected)} frames")
        return selected

