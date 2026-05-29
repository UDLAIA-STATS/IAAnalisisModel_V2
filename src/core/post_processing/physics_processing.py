import logfire
import numpy as np
from sqlmodel import Session

from src.core.repository.depth_history_repository import DepthRepository
from src.entities.services.physics_processing_base import PhysicsCalculatorBase
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.core.vision.pixels_converter import pixel_conversion_handler


class PhysicsProcessing(PhysicsCalculatorBase):
    def process(self, match_id: int, fps: float, session: Session):
        states = self.get_players(match_id, session)

        if not states:
            return
        for _, player_states in states.items():
            prev_state = None
            for state in player_states:
                if prev_state is None:
                    prev_state = state
                    continue

                if prev_state.frame_number == state.frame_number:
                    prev_state = state
                    continue

                gap = state.frame_number - prev_state.frame_number

                if gap > fps * 3:
                    continue

                # actual_constant = pixel_conversion_handler.get_current_conversion()

                # actual_constant_state = DepthRepository.get_depth_by_player(
                #     match_id, state.player_id, state.frame_number, session
                # )

                # if actual_constant_state is not None:
                #     actual_constant = actual_constant_state.constant

                actual_constant = 1

                xo = self._bbox_to_center(
                    [prev_state.x1, prev_state.y1, prev_state.x2, prev_state.y2]
                )
                xf = self._bbox_to_center([state.x1, state.y1, state.x2, state.y2])

                prev_state.dx = float(xo[0])
                prev_state.dy = float(xo[1])

                state.dx = float(xf[0])
                state.dy = float(xf[1])

                xo = xo
                xf = xf
                delta_t = state.timestamp - prev_state.timestamp
                logfire.info(f"Delta t: {delta_t} result from {prev_state.timestamp} in frame {prev_state.frame_number} to {state.timestamp} in frame {state.frame_number}")
                distance, delta_x = self.calculate_distance(xo, xf)

                vo, vf, speed_ms, acceleration_ms, ax, ay = self.calculate_kinematics(
                    delta_x * actual_constant, delta_t, np.asarray([prev_state.vx, prev_state.vy])
                )


                state.dx_meters = float(delta_x[0]) * actual_constant
                state.dy_meters = float(delta_x[1]) * actual_constant

                state.delta_x = float(delta_x[0])
                state.delta_y = float(delta_x[1])
                state.distance_meters = float(distance) * actual_constant

                state.ax = float(ax)
                state.ay = float(ay)
                state.acceleration = float(acceleration_ms)
                state.speed_kmh = float(speed_ms) * 3.6

                prev_state.vx = float(vo[0])
                prev_state.vy = float(vo[1])

                state.vx = float(vf[0])
                state.vy = float(vf[1])

                PlayerStatesRepository.update_state(state, session)
                PlayerStatesRepository.update_state(prev_state, session)

                prev_state = state

        session.commit()


physics_procesor = PhysicsProcessing()
