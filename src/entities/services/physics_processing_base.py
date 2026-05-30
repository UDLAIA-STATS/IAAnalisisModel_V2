import math
from typing import Dict, List, Tuple

import logfire
import numpy as np
from sqlmodel import Session

from src.core.repository.player_states_repository import PlayerStatesRepository
from src.core.repository.player_repository import PlayerRepository
from src.entities.models.soccer.player_model import PlayerState


class PhysicsCalculatorBase:
    MAX_SPEED = 14.0
    MAX_ACCELERATION = 12.0
    MAX_FRAME_GAP = 40
    SPEED_SMOOTHING = 0.15
    
    def calculate_distance(
        self, xo: np.ndarray, xf: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        delta_x = xf - xo
        return float(np.linalg.norm(delta_x)), delta_x

    def calculate_kinematics(
        self, delta_x: np.ndarray, delta_t: float, prev_speed: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        if delta_t == 0:
            return prev_speed, prev_speed, 0.0, 0.0, 0.0, 0.0

        vo = prev_speed
        vf = delta_x / delta_t

        acceleration = (vf - vo) / delta_t

        ax, ay = acceleration
        speed_ms = float(np.linalg.norm(vf))

        if speed_ms > self.MAX_SPEED:
            vf = (
                self.SPEED_SMOOTHING * vf + (1 - self.SPEED_SMOOTHING) * prev_speed
            )

            speed_ms = float(np.linalg.norm(vf) * 3.6)
            logfire.info(f"[PhysicsCalculator] Speed is too high, new speed: {speed_ms} m/s")

        acceleration_ms = float(np.linalg.norm(acceleration))

        if acceleration_ms > self.MAX_ACCELERATION:
            acceleration_ms = self.MAX_ACCELERATION

        return vo, vf, speed_ms, acceleration_ms, float(ax), float(ay)

    def get_players(
        self, match_id: int, session: Session
    ) -> Dict[int, List[PlayerState]]:
        players = PlayerRepository.get_players_by_match_id(match_id, session)
        states: Dict[int, List[PlayerState]] = {}

        if len(players) == 0:
            logfire.error(
                f"[PhysicsCalculator] No players found for match {match_id} in database"
            )
            return states

        for player in players:
            if not player.states:
                continue

            if player.track_id not in states:
                states[player.track_id] = []

            states[player.track_id].extend(player.states)

        for value in states.values():
            value.sort(key=lambda state: state.frame_number)

        self.validate_player_frames(states, session)
        self._calculate_coords(states, session)
        
        cleared_states = {}

        for values in states.values():
            prev_state = None
            for value in values:
                if prev_state is None:
                    prev_state = value
                    continue

                if prev_state is None:
                    logfire.fatal("[PhysicsCalculator] prev_state is None")

                if prev_state.dx == value.dx and prev_state.dy == value.dy:
                    continue

                cleared_states[value.player_id] = values
                prev_state = value

            values.sort(key=lambda state: state.frame_number)

        return states

    def _calculate_coords(self, values: Dict[int, List[PlayerState]], session: Session):
        for _, states in values.items():
            for state in states:
                bbox = [state.x1, state.y1, state.x2, state.y2]
                dx, dy = self._bbox_to_center(bbox)
                state.dx = float(dx)
                state.dy = float(dy)
                PlayerStatesRepository.update_state(state, session)

    def _generate_middle_state(
    self,
    state_a: PlayerState,
    state_b: PlayerState,
    t: float,
    session: Session
    ) -> PlayerState:
        """
        Creates a synthetic interpolated state between two states
        whose frame gap exceeds MAX_FRAME_GAP.
        """
        mid_frame = int(state_a.frame_number + t * (state_b.frame_number - state_a.frame_number))

        bbox_a = [state_a.x1, state_a.y1, state_a.x2, state_a.y2]
        bbox_b = [state_b.x1, state_b.y1, state_b.x2, state_b.y2]

        center_a = self._bbox_to_center(bbox_a)
        center_b = self._bbox_to_center(bbox_b)
        mid_center = center_a + t * (center_b - center_a)

        aw = bbox_a[2] - bbox_a[0]
        ah = bbox_a[3] - bbox_a[1]
        bw = bbox_b[2] - bbox_b[0]
        bh = bbox_b[3] - bbox_b[1]
        mw = aw + t * (bw - aw)
        mh = ah + t * (bh - ah)

        cx, cy = float(mid_center[0]), float(mid_center[1])
        mid_timestamp = float(state_a.timestamp + t * (state_b.timestamp - state_a.timestamp))
        mid_confidence = (state_a.confidence + state_b.confidence) / 2

        new_state = PlayerState(
            frame_number=mid_frame,
            x1=cx - mw / 2,
            y1=cy - mh / 2,
            x2=cx + mw / 2,
            y2=cy + mh / 2,
            player_id=state_a.player_id,
            timestamp=mid_timestamp,
            confidence=mid_confidence,
        )
        session.add(new_state)
        session.flush()
        return new_state


    def _insert_middle_states(
        self, states: List[PlayerState], session: Session
    ) -> List[PlayerState]:
        """
        Iterates over a sorted state list and recursively inserts
        synthetic middle states wherever the frame gap > MAX_FRAME_GAP,
        until every consecutive gap is within the allowed limit.
        """
        changed = True
        while changed:
            changed = False
            result: List[PlayerState] = []
            for i, state in enumerate(states):
                result.append(state)
                if i == len(states) - 1:
                    break
                gap = states[i + 1].frame_number - state.frame_number
                if gap > self.MAX_FRAME_GAP:
                    num_segments = math.ceil(gap / self.MAX_FRAME_GAP)
                    if num_segments == 0 and gap == self.MAX_FRAME_GAP:
                        num_segments = 1
                        logfire.info(f"[PhysicsCalculator] Frame gap {gap} too large to interpolate. Creating middle segment for default...")

                    for j in range(1, num_segments):
                        t = j / num_segments
                        middle = self._generate_middle_state(state, states[i + 1], t, session)
                        result.append(middle)
                    changed = True
            states = result
        return states


    def validate_player_frames(
        self, states: Dict[int, List[PlayerState]], session: Session
    ) -> None:
        """
        Sorts each player's states and fills large frame gaps with
        synthetic interpolated states so downstream physics calculations
        never see a delta-t that would imply speed > MAX_SPEED.
        """
        for track_id, player_states in states.items():
            player_states.sort(key=lambda s: s.frame_number)
            states[track_id] = self._insert_middle_states(player_states, session)

    def _bbox_to_center(self, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        return np.asarray([cx, cy])
