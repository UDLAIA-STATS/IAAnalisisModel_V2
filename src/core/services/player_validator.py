import math
from typing import Dict, List, Tuple
import logfire
from skimage import color

import numpy as np
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
    MAX_DELTA: int = 60
    MIN_FRAMES = 10
    COLOR_DISTANCE = 5

    def validate(self, match_id: int, total_frames: int, session: Session):
        logfire.info(f"[PlayerValidator] Validating match {match_id} with {total_frames} frames")
        states = PlayerStatesRepository.get_states_by_frame_range(
            match_id=match_id,
            min_frame=1,
            max_frame=total_frames,
            session=session
        )
        logfire.info(f"[PlayerValidator] Validating {len(states)} frames")
        players: Dict[int, List[PlayerState]] = {}
        players_incorrect: Dict[int, List[PlayerState]] = {}

        for state in states:
            frame_number = state.frame_number
            if state.player.track_id <= 22:
                if state.frame_number not in players:
                    players[frame_number] = [state]
                else:
                    players[frame_number].append(state)
            else:
                if state.frame_number not in players_incorrect:
                    players_incorrect[frame_number] = [state]
                else:
                    players_incorrect[frame_number].append(state)

        logfire.info(f"[PlayerValidator] Validating {len(players)} correct players and {len(players_incorrect)} incorrect players")
        player_centroids: Dict[int, List[PlayerCentroid]] = {}
        player_incorrect_centroids: Dict[int, List[PlayerCentroid]] = {}

        if len(players) == 0 or len(players_incorrect) == 0:
            logfire.info("[PlayerValidator] No players to validate")
            return

        for number, correct_players in players.items():
            for player in correct_players:
                cx, cy = self.calculate_centroid(player)
                centroid = PlayerCentroid(
                    color=player.player.team_color,
                    conf=player.confidence,
                    tracker_id=player.player.track_id,
                    player_id=player.player_id,
                    cx=cx,
                    cy=cy,
                    frame_number=player.frame_number
                )

                if number not in player_centroids:
                    player_centroids[number] = [centroid]
                else:
                    player_centroids[number].append(centroid)
        
        for number, incorrect_players in players_incorrect.items():
            for state in incorrect_players:
                cx, cy = self.calculate_centroid(state)
                centroid = PlayerCentroid(
                    color=state.player.team_color,
                    conf=state.confidence,
                    player_id=state.player_id,
                    tracker_id=state.player.track_id,
                    cx=cx,
                    cy=cy,
                    frame_number=state.frame_number
                )

                if number not in player_incorrect_centroids:
                    player_incorrect_centroids[number] = [centroid]
                else:
                    player_incorrect_centroids[number].append(centroid)

        matches: Dict[int, List[Tuple[PlayerCentroid, PlayerCentroid]]] = {}

        for frame in range(1, total_frames, self.MIN_FRAMES):
            if frame not in player_incorrect_centroids and frame not in player_centroids:
                continue

            correct_players = player_centroids[frame]
            incorrect_players = self.get_incorrect_players(player_incorrect_centroids, frame, total_frames)

            for incorrect_player in incorrect_players:
                candidates: List[Tuple[PlayerCentroid, float]] = []
                
                for correct_player in correct_players:
                    distance = self.calculate_distance((correct_player.cx, correct_player.cy), (incorrect_player. cx, incorrect_player.cy))
                    color_distance = self.calculate_color_distance(correct_player.color, incorrect_player.color)
                    logfire.info(f"[PlayerValidator] Calculated color distance: {color_distance}")

                    if distance < self.MAX_DELTA and color_distance < self.COLOR_DISTANCE:
                        candidates.append((correct_player, distance))
                
                candidates.sort(key=lambda x: x[1])

                if candidates:
                    correct_player, _ = candidates[0]

                    if incorrect_player.player_id not in matches:
                        matches[incorrect_player.player_id] = [(correct_player, incorrect_player)]
                    else:
                        matches[incorrect_player.player_id].append((correct_player, incorrect_player))

        logfire.info(f"[PlayerValidator] Validating {len(matches)} incorrect players")
        logfire.info(f"[PlayerValidator] Validating matches keys: {matches.keys()}")
        to_merge_players: List[Tuple[int, int]] = []

        for incorrect_player_id, candidate_list in matches.items():
            best_candidate = candidate_list[0][0]
            best_distance = 0
            best_color_distance = 0
            for correct_player, incorrect_player in candidate_list:
                diff_frames = math.fabs(correct_player.frame_number - incorrect_player.frame_number)

                if diff_frames > 25:
                    continue

                dominance = np.sum(np.array([1 for candidate, _ in candidate_list if candidate.player_id == correct_player.player_id]))
                distance = self.calculate_distance((correct_player.cx, correct_player.cy), (incorrect_player.cx, incorrect_player.cy))
                color_distance = self.calculate_color_distance(correct_player.color, incorrect_player.color)

                if dominance > len(candidate_list) // 2:
                    best_candidate = correct_player
                    logfire.info(f"[PlayerValidator] New best candidate: {best_candidate}")
                    break

                if (distance < best_distance and color_distance < best_color_distance):
                    best_candidate = correct_player
                    best_distance = distance
                    best_color_distance = color_distance
                    logfire.info(f"[PlayerValidator] New best candidate: {best_candidate}")
                    continue
            
            to_merge_players.append((incorrect_player_id, best_candidate.player_id))

        logfire.info(f"[PlayerValidator] Merging {len(to_merge_players)} incorrect players")
        logfire.info(f"[PlayerValidator] Merging players: {to_merge_players}")

        for incorrect_player_id, correct_player_id in to_merge_players:
            PlayerStatesRepository.merge_states(correct_player_id, incorrect_player_id, session)

        session.commit()

    def get_incorrect_players(self, players: Dict[int, List[PlayerCentroid]], actual_frame: int, total_frames: int):
        incorrect_players: List[PlayerCentroid] = []

        for frame in range(actual_frame, actual_frame + self.MIN_FRAMES, int(self.MIN_FRAMES / 3)):
            if frame in players:
                incorrect_players.extend(players[frame])

        logfire.info(f"[PlayerValidator] Incorrect players selected: {len(incorrect_players)}")
        return incorrect_players

    def calculate_distance(self, centroid_a: tuple, centroid_b: tuple):
        cx1, cy1 = centroid_a
        cx2, cy2 = centroid_b

        return float(np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))

    def calculate_centroid(self, player: PlayerState):
        cx = (player.x1 + player.x2) / 2
        cy = (player.y1 + player.y2) / 2
        return cx, cy

    def calculate_color_distance(self, color_a: str, color_b: str):
        rgb_a = np.array([int(x.strip()) for x in color_a.split(",")], dtype=np.float32)
        normalized_a = rgb_a / 255.0
        input_a = np.reshape(normalized_a, (1, 1, 3))

        rgb_b = np.array([int(x.strip()) for x in color_b.split(",")], dtype=np.float32)
        normalized_b = rgb_b / 255.0
        input_b = np.reshape(normalized_b, (1, 1, 3))

        lab_a = color.rgb2lab(input_a)
        lab_b = color.rgb2lab(input_b)

        l, a, b = lab_a[0][0]
        l2, a2, b2 = lab_b[0][0]

        return float(np.sqrt((l2 - l) ** 2 + (a2 - a) ** 2 + (b2 - b) ** 2))
