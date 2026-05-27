import math
from typing import Dict, List, Tuple, Set
import logfire
import numpy as np
from skimage import color
from collections import defaultdict

from pydantic import BaseModel
from sqlmodel import Session

from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.soccer.player_model import PlayerState


class PlayerCentroid(BaseModel):
    player_id: int
    tracker_id: int
    cx: float
    cy: float
    color: str
    conf: float
    frame_number: int


class TrackSummary(BaseModel):
    player_id: int
    tracker_id: int
    frame_start: int
    frame_end: int
    detection_count: int
    avg_color: str
    first_cx: float
    first_cy: float
    last_cx: float
    last_cy: float


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


class PlayerValidator:
    # ── spatial / color hard limits ──────────────────────────────────────────
    MAX_DISTANCE = 130          # max px between re-entry positions
    MAX_COLOR_DISTANCE = 40     # max ΔE (Lab) for color gate
    FRAME_STEP = 20             # kept for backward-compat helper

    # ── composite score weights & threshold ──────────────────────────────────
    # score = W_COLOR·color_dist + W_POSITION·(pos_dist/10) + W_GAP·(gap/100)
    # lower score = better match; pairs above MERGE_SCORE_THRESHOLD are skipped
    W_COLOR: float = 0.5
    W_POSITION: float = 0.3
    W_GAP: float = 0.2
    MERGE_SCORE_THRESHOLD: float = 20.0

    # ── ghost track pruning ───────────────────────────────────────────────────
    # tracks with fewer than this many detections are almost always occlusion
    # artifacts; they are scored first so they don't pollute the main pass
    GHOST_TRACK_THRESHOLD: int = 10

    def validate(self, match_id: int, total_frames: int, session: Session) -> None:
        logfire.info(f"[PlayerValidator] Starting validation for match {match_id}")

        states = PlayerStatesRepository.get_states_by_frame_range(
            match_id=match_id,
            min_frame=1,
            max_frame=total_frames,
            session=session,
        )

        correct_states: Dict[int, List[PlayerState]] = defaultdict(list)
        incorrect_states: Dict[int, List[PlayerState]] = defaultdict(list)

        for state in states:
            if state.player.track_id <= 22:
                correct_states[state.frame_number].append(state)
            else:
                incorrect_states[state.frame_number].append(state)

        if not correct_states or not incorrect_states:
            logfire.info("[PlayerValidator] No data to validate")
            return

        correct_centroids = self._build_centroids(correct_states)
        incorrect_centroids = self._build_centroids(incorrect_states)

        correct_summaries = self._build_track_summaries(correct_centroids)
        incorrect_summaries = self._build_track_summaries(incorrect_centroids)

        ghost_ids: Set[int] = {
            pid for pid, s in incorrect_summaries.items()
            if s.detection_count <= self.GHOST_TRACK_THRESHOLD
        }
        logfire.info(
            f"[PlayerValidator] Correct tracks: {len(correct_summaries)} | "
            f"Incorrect tracks: {len(incorrect_summaries)} | "
            f"Ghost tracks: {ghost_ids}"
        )

        # ── score every valid (incorrect, correct) candidate pair ─────────────
        # A pair is valid only when:
        #   1. No frame overlap exists (hard constraint — same player can't be
        #      in two places at once)
        #   2. Color distance is below MAX_COLOR_DISTANCE
        #   3. Spatial re-entry distance is below 2 × MAX_DISTANCE
        #
        # Ghosts are prepended so the sort naturally resolves them first.

        scored_candidates: List[Tuple[float, int, int]] = []  # (score, incorrect_id, correct_id)

        for incorrect_id, inc in incorrect_summaries.items():
            is_ghost = incorrect_id in ghost_ids

            for correct_id, cor in correct_summaries.items():

                # ── hard constraint: temporal non-overlap ─────────────────────
                if self._has_frame_overlap(cor, inc):
                    continue

                color_dist = self.calculate_color_distance(inc.avg_color, cor.avg_color)
                if color_dist >= self.MAX_COLOR_DISTANCE:
                    continue

                # ── spatial re-entry: compare end of earlier track to start
                #    of the later one, not arbitrary frame centroids ──────────
                if cor.frame_end < inc.frame_start:
                    early, late = cor, inc
                else:
                    early, late = inc, cor

                gap = late.frame_start - early.frame_end
                pos_dist = self.calculate_distance(
                    (early.last_cx, early.last_cy),
                    (late.first_cx, late.first_cy),
                )

                if pos_dist >= self.MAX_DISTANCE * 2:
                    continue

                score = self._composite_score(color_dist, pos_dist, gap)

                # Bias ghosts slightly toward the front of the sorted list so
                # they are consumed before longer tracks compete for the same
                # correct_id.
                if is_ghost:
                    score *= 0.9

                scored_candidates.append((score, incorrect_id, correct_id))
                logfire.info(
                    f"[PlayerValidator] Candidate {incorrect_id}↔{correct_id}: "
                    f"score={score:.2f}  color_dist={color_dist:.1f}  "
                    f"pos_dist={pos_dist:.1f}  gap={gap}"
                )

        # ── greedy merge: consume pairs in score order ─────────────────────
        # Each incorrect_id is merged at most once; multiple correct_ids that
        # compete for the same incorrect_id are resolved by score ordering.
        # Union-Find propagates transitive chains automatically.

        scored_candidates.sort(key=lambda x: x[0])

        uf = UnionFind()
        merged_incorrect: Set[int] = set()

        for score, incorrect_id, correct_id in scored_candidates:
            if score > self.MERGE_SCORE_THRESHOLD:
                break
            if incorrect_id in merged_incorrect:
                continue

            uf.union(correct_id, incorrect_id)
            merged_incorrect.add(incorrect_id)
            logfire.info(
                f"[PlayerValidator] Scheduling merge {incorrect_id} → {correct_id} "
                f"(score={score:.2f})"
            )

        # ── execute merges derived from union-find groups ─────────────────
        # For each connected component, every non-root member is merged into
        # the root (canonical) ID.  This correctly handles chains such as
        # correct_id=3 ← incorrect_id=32 ← incorrect_id=38.

        merge_groups = uf.groups()
        total_merged = 0

        for root, members in merge_groups.items():
            for member in members:
                if member == root:
                    continue
                logfire.info(f"[PlayerValidator] Merging player_id={member} → {root}")
                PlayerStatesRepository.merge_states(root, member, session)
                total_merged += 1

        session.commit()
        logfire.info(
            f"[PlayerValidator] Validation completed. "
            f"Merged {total_merged} tracks across {len(merge_groups)} groups."
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _has_frame_overlap(self, a: TrackSummary, b: TrackSummary) -> bool:
        """Return True if the two tracks are active at the same frame."""
        return max(a.frame_start, b.frame_start) <= min(a.frame_end, b.frame_end)

    def _composite_score(self, color_dist: float, pos_dist: float, gap: int) -> float:
        """Lower score = better match."""
        return (
            self.W_COLOR * color_dist
            + self.W_POSITION * (pos_dist / 10)
            + self.W_GAP * (gap / 100)
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

    # ── public geometry / color utilities ────────────────────────────────────

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
