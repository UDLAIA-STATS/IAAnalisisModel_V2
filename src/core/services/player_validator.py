import math
from typing import Dict, List, Tuple
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


class PlayerValidator:
    MAX_DISTANCE = 130
    MAX_COLOR_DISTANCE = 40
    FRAME_STEP = 20
    MAX_FRAME_GAP = 45

    def validate(self, match_id: int, total_frames: int, session: Session):
        logfire.info(f"[PlayerValidator] Starting validation for match {match_id}")

        states = PlayerStatesRepository.get_states_by_frame_range(
            match_id=match_id,
            min_frame=1,
            max_frame=total_frames,
            session=session
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

        logfire.info(f"[PlayerValidator] Correct tracks: {len(correct_centroids)} frames | "
                    f"Incorrect tracks: {len(incorrect_centroids)} frames")

        matches: Dict[int, List[Tuple[PlayerCentroid, PlayerCentroid, float, float]]] = defaultdict(list)

        for frame in range(1, total_frames + 1, self.FRAME_STEP):
            correct_in_frame = correct_centroids.get(frame, [])
            if not correct_in_frame:
                continue

            incorrect_candidates = self._get_incorrect_near_frame(incorrect_centroids, frame, total_frames, self.FRAME_STEP)

            for incorrect in incorrect_candidates:
                for correct in correct_in_frame:
                    dist = self.calculate_distance((correct.cx, correct.cy), (incorrect.cx, incorrect.cy))
                    color_dist = self.calculate_color_distance(correct.color, incorrect.color)
                    logfire.info(f"[PlayerValidator] Dist: {dist} | Color dist: {color_dist}")

                    if dist < self.MAX_DISTANCE and color_dist < self.MAX_COLOR_DISTANCE:
                        matches[incorrect.player_id].append((correct, incorrect, dist, color_dist))

        to_merge: List[Tuple[int, int]] = []

        for incorrect_id, candidates in matches.items():
            if not candidates:
                continue

            groups: Dict[int, List[Tuple[PlayerCentroid, PlayerCentroid, float, float]]] = {}
            best_correct_id = None
            best_dist = math.inf
            best_color_dist = math.inf
            for correct, incorrect, dist, color_dist in candidates:
                if correct.player_id not in groups:
                    groups[correct.player_id] = []
                groups[correct.player_id].append((correct, incorrect, dist, color_dist))


            for correct_id, group in groups.items():
                avg_dist = sum(distance for _, _, distance, _ in group) / len(group)
                avg_color_dist = sum(cd for _, _, _, cd in group) / len(group)
                max_gap = max(abs(c.frame_number - i.frame_number) for c, i, _, _ in group)

                if max_gap > self.MAX_FRAME_GAP:
                    logfire.info(f"[PlayerValidator] Max gap: {max_gap} for player {correct_id}")
                    continue

                logfire.info(f"[PlayerValidator] Avg dist: {avg_dist} | Avg color dist: {avg_color_dist}")

                if avg_dist < self.MAX_DISTANCE and avg_dist < best_dist: #and avg_color_dist < self.MAX_COLOR_DISTANCE and avg_color_dist < best_color_dist:
                    best_dist = avg_dist
                    best_color_dist = avg_color_dist
                    best_correct_id = correct_id


            if best_correct_id is not None:
                to_merge.append((incorrect_id, best_correct_id))

        logfire.info(f"[PlayerValidator] Merging {len(to_merge)} incorrect track IDs")
        for incorrect_id, correct_id in to_merge:
            PlayerStatesRepository.merge_states(correct_id, incorrect_id, session)

        session.commit()
        logfire.info(f"[PlayerValidator] Validation completed. Merged {len(to_merge)} players.")

    def _build_centroids(self, states_dict: Dict[int, List[PlayerState]]) -> Dict[int, List[PlayerCentroid]]:
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
                    frame_number=frame
                )
                centroids[frame].append(centroid)

        return centroids

    def _get_incorrect_near_frame(
            self, incorrect_centroids: Dict[int, List[PlayerCentroid]], 
            actual_frame: int, total_frames: int, search_range: int = 20) -> List[PlayerCentroid]:
        """Get incorrect players from nearby frames for more robust matching"""
        incorrect_players: List[PlayerCentroid] = []
        stop = actual_frame + search_range

        if stop > total_frames:
            stop = total_frames

        for frame in range(actual_frame, stop):
            if frame in incorrect_centroids:
                incorrect_players.extend(incorrect_centroids[frame])

        return incorrect_players

    def calculate_distance(self, centroid_a: tuple, centroid_b: tuple) -> float:
        cx1, cy1 = centroid_a
        cx2, cy2 = centroid_b
        return float(np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))

    def calculate_centroid(self, player: PlayerState):
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