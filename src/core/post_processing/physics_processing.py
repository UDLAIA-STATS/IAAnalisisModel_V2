import numpy as np
from sqlmodel import Session

from entities.services.physics_processing_model import PhysicsCalculatorBase


class PhysicsProcessing(PhysicsCalculatorBase):
    def process(self, match_id: int, constant: float, session: Session):
        states = self.get_players(match_id, session)

        if not states:
            return
        
        for track_id, player_states in states.items():
            prev_state = None
            for state in player_states:
                if prev_state is None:
                    prev_state = state
                    continue

                xo = self._bbox_to_center([prev_state.x1, prev_state.y1, prev_state.x2, prev_state.y2])
                xf = self._bbox_to_center([state.x1, state.y1, state.x2, state.y2])
                delta_t = state.timestamp - prev_state.timestamp
                distance, delta_x = self.calculate_distance(xo, xf)
                vo, vf, acceleration_ms, speed_kmh = self.calculate_speed(delta_x, delta_t)

                prev_state.dx = xo[0]
                prev_state.dy = xo[1]
                
                state.dx = xf[0]
                state.dy = xf[1]

                state.delta_x = delta_x[0]
                state.delta_y = delta_x[1]

                state.distance = distance
                state.acceleration = acceleration_ms
                state.speed_kmh = speed_kmh
                state.vx = vo[0]
                state.vy = vo[1]