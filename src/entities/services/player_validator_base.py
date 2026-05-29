from typing import Dict, List, Tuple
import numpy as np
from skimage import color
from collections import defaultdict

from src.entities.models.soccer.player_model import PlayerState
from src.entities.models.app.player_validator_models import PlayerCentroid, TrackSummary

class UnionFind:
    """
    Union-Find (disjoint set) for transitive merge propagation.
    Ensures that if A→B and B→C are both valid merges, all three
    collapse to a single canonical ID rather than being applied
    as two independent pairs.
    """

    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # union by rank keeps the tree shallow
        if self.rank.get(rx, 0) < self.rank.get(ry, 0):
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank.get(rx, 0) == self.rank.get(ry, 0):
            self.rank[rx] = self.rank.get(rx, 0) + 1

    def groups(self) -> Dict[int, List[int]]:
        """Return {root_id: [all member ids]} for every group with >1 member."""
        result: Dict[int, List[int]] = defaultdict(list)
        for x in self.parent:
            result[self.find(x)].append(x)
        return {root: members for root, members in result.items() if len(members) > 1}


class PlayerValidatorBase:
    # ── spatial / color hard limits ──────────────────────────────────────────
    MAX_DISTANCE = 130          # max px between re-entry positions
    MAX_COLOR_DISTANCE = 40     # max ΔE (Lab) for color gate
    FRAME_STEP = 20             # kept for backward-compat helper
    MAX_FRAME_GAP = 80

    # ── composite score weights & threshold ──────────────────────────────────
    # score = W_COLOR·color_dist + W_POSITION·(pos_dist/10) + W_GAP·(gap/100)
    # lower score = better match; pairs above MERGE_SCORE_THRESHOLD are skipped
    W_COLOR: float = 0.35
    W_POSITION: float = 0.55
    W_GAP: float = 0.1
    MERGE_SCORE_THRESHOLD: float = 20.0

    # ── ghost track pruning ───────────────────────────────────────────────────
    # tracks with fewer than this many detections are almost always occlusion
    # artifacts; they are scored first so they don't pollute the main pass
    GHOST_TRACK_THRESHOLD: int = 10

    def _has_frame_overlap(self, summary_a: TrackSummary, summary_b: TrackSummary) -> bool:
        """Return True if the two tracks are active at the same frame."""
        return max(summary_a.frame_start, summary_b.frame_start) <= min(summary_a.frame_end, summary_b.frame_end)

    def _composite_score(self, color_dist: float, pos_dist: float, gap: int) -> float:
        """Lower score = better match."""
        color_score = color_dist / self.MAX_COLOR_DISTANCE
        position_score = pos_dist / self.MAX_DISTANCE
        gap_score = gap / self.GHOST_TRACK_THRESHOLD

        return (
            self.W_COLOR * color_score
            + self.W_POSITION * position_score
            + self.W_GAP * gap_score
        )

    def _build_track_summaries(
        self, centroids: Dict[int, List[PlayerCentroid]]
    ) -> Dict[int, TrackSummary]:
        """
        Collapse the per-frame centroid dict into one TrackSummary per player.

        Averaging the color over all detections makes the color signal much
        more robust than using a single frame, which can be affected by motion
        blur or partial occlusion.
        """
        per_player: Dict[int, List[PlayerCentroid]] = defaultdict(list)
        for frame_centroids in centroids.values():
            for c in frame_centroids:
                per_player[c.player_id].append(c)

        summaries: Dict[int, TrackSummary] = {}
        for player_id, detections in per_player.items():
            detections.sort(key=lambda c: c.frame_number)
            first = detections[0]
            last = detections[-1]

            color_matrix = np.array(
                [[int(v.strip()) for v in d.color.split(",")] for d in detections],
                dtype=np.float32,
            )
            avg_rgb = color_matrix.mean(axis=0).astype(int)
            avg_color_str = f"{avg_rgb[0]},{avg_rgb[1]},{avg_rgb[2]}"

            summaries[player_id] = TrackSummary(
                player_id=player_id,
                tracker_id=first.tracker_id,
                frame_start=first.frame_number,
                frame_end=last.frame_number,
                detection_count=len(detections),
                avg_color=avg_color_str,
                first_cx=first.cx,
                first_cy=first.cy,
                last_cx=last.cx,
                last_cy=last.cy,
            )

        return summaries

    def _build_centroids(
        self, states_dict: Dict[int, List[PlayerState]]
    ) -> Dict[int, List[PlayerCentroid]]:
        centroids: Dict[int, List[PlayerCentroid]] = defaultdict(list)

        for frame, state_list in states_dict.items():
            for state in state_list:
                cx, cy = self.calculate_centroid(state)
                centroid = PlayerCentroid(
                    player_id=state.player_id,
                    tracker_id=state.player.track_id,
                    cx=cx,
                    cy=cy,
                    color=state.player.team_color,
                    conf=state.confidence,
                    frame_number=frame,
                )
                centroids[frame].append(centroid)

        return centroids

    def _get_incorrect_near_frame(
        self,
        incorrect_centroids: Dict[int, List[PlayerCentroid]],
        actual_frame: int,
        total_frames: int,
        search_range: int = 20,
    ) -> List[PlayerCentroid]:
        """
        Return incorrect-track centroids from frames within ±search_range of
        actual_frame.  The original only searched forward; this searches both
        directions so that an incorrect track that appears slightly before a
        correct one can still be matched.
        """
        result: List[PlayerCentroid] = []
        start = max(1, actual_frame - search_range)
        stop = min(total_frames, actual_frame + search_range)

        for frame in range(start, stop + 1):
            if frame in incorrect_centroids:
                result.extend(incorrect_centroids[frame])

        return result

    def calculate_distance(self, centroid_a: tuple, centroid_b: tuple) -> float:
        cx1, cy1 = centroid_a
        cx2, cy2 = centroid_b
        return float(np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))

    def calculate_centroid(self, player: PlayerState) -> Tuple[float, float]:
        cx = (player.x1 + player.x2) / 2
        cy = (player.y1 + player.y2) / 2
        return cx, cy

    def calculate_color_distance(self, color_a: str, color_b: str) -> float:
        rgb_a = np.array([int(x.strip()) for x in color_a.split(",")], dtype=np.float32) / 255.0
        rgb_b = np.array([int(x.strip()) for x in color_b.split(",")], dtype=np.float32) / 255.0

        lab_a = color.rgb2lab(rgb_a.reshape(1, 1, 3))
        lab_b = color.rgb2lab(rgb_b.reshape(1, 1, 3))

        l1, a1, b1 = lab_a[0][0]
        l2, a2, b2 = lab_b[0][0]

        return float(np.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2))
