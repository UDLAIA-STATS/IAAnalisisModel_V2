import logfire
import numpy as np
from sqlmodel import Session

from src.core.vision.homography_interpolator import HomographyInterpolator
from src.entities.services.physics_processing_base import PhysicsCalculatorBase
from src.core.repository.player_states_repository import PlayerStatesRepository


class PhysicsProcessing(PhysicsCalculatorBase):
    def process(self, match_id: int, fps: float, session: Session):
        logfire.info(f"[PhysicsProcessing] Processing match {match_id}")
        states = self.get_players(match_id, session)

        if not states:
            logfire.error(
                f"[PhysicsProcessing] No players found for match {match_id},"
                " cannot project to field coordinates"
            )
            return

        interpolator = HomographyInterpolator.from_match(match_id, session)


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
                    prev_state = state
                    continue


                # actual_constant = pixel_conversion_handler.get_current_conversion()

                # actual_constant_state = DepthRepository.get_depth_by_player(
                #     match_id, state.player_id, state.frame_number, session
                # )

                # if actual_constant_state is not None:
                #     actual_constant = actual_constant_state.constant

                actual_constant = 1

                if not interpolator:
                    curr_pos = np.array([state.dx, state.dy])
                    prev_pos = np.array([prev_state.dx, prev_state.dy])
                else:
                    curr_pos = interpolator.project(
                        state.x1,
                        state.y1,
                        state.x2,
                        state.y2,
                        state.frame_number,
                    )
                    state.dx_meters = float(curr_pos.x_meters)
                    state.dy_meters = float(curr_pos.y_meters)

                    prev_pos = interpolator.project(
                        prev_state.x1,
                        prev_state.y1,
                        prev_state.x2,
                        prev_state.y2,
                        prev_state.frame_number,
                    )
                    prev_state.dx_meters = float(prev_pos.x_meters)
                    prev_state.dy_meters = float(prev_pos.y_meters)

                # if prev_state.dx_meters is not None and state.dx_meters is not None:
                #     jump_distance = np.sqrt(
                #         (state.dx_meters - prev_state.dx_meters) ** 2
                #         + (state.dy_meters - prev_state.dy_meters) ** 2
                #     )
                #     max_possible_jump = 10.0
                #     if jump_distance > max_possible_jump:
                #         logfire.warning(
                #             f"[PhysicsProcessing] Id Switched Player {state.player.track_id} jumped {jump_distance} meters in {gap} frames"
                #         )
                #         prev_state = state
                #         continue

                delta_t = state.timestamp - prev_state.timestamp
                # logfire.info(f"Delta t: {delta_t} result from {prev_state.timestamp} in frame {prev_state.frame_number} to {state.timestamp} in frame {state.frame_number}")
                
                if prev_state.dx_meters is None or prev_state.dy_meters is None:
                    logfire.info(f"[PhysicsProcessing] No dx or dy meters for player {prev_state.player_id} in frame {prev_state.frame_number}")
                    xo = np.array([prev_state.dx, prev_state.dy])
                else:
                    xo = np.array([prev_state.dx_meters, prev_state.dy_meters])

                if state.dx_meters is None or state.dy_meters is None:
                    logfire.info(f"[PhysicsProcessing] No dx or dy meters for player {state.player_id} in frame {state.frame_number}")
                    xf = np.array([state.dx, state.dy])
                else:
                    xf = np.array([state.dx_meters, state.dy_meters])

                distance, delta_x = self.calculate_distance(xo, xf)

                vo, vf, speed_ms, acceleration_ms, ax, ay = self.calculate_kinematics(
                    delta_x,
                    delta_t,
                    np.asarray([prev_state.vx, prev_state.vy]),
                )

                state.delta_x = float(delta_x[0])
                state.delta_y = float(delta_x[1])
                state.distance_meters = float(distance)

                state.ax = float(ax)
                state.ay = float(ay)
                state.acceleration = float(acceleration_ms)
                speed_kmh = float(speed_ms) * 3.6

                if speed_kmh > 20:
                    # logfire.warning(
                    #     f"[PhysicsProcessing] Player {state.player_id} moved too fast {speed_kmh} km/h in frame {state.frame_number}"
                    # )
                    speed_kmh = 12.5
                    vf = vf / np.linalg.norm(vf) * speed_kmh if np.linalg.norm(vf) > 0 else vf

                state.speed_kmh = speed_kmh

                prev_state.vx = float(vo[0])
                prev_state.vy = float(vo[1])

                state.vx = float(vf[0])
                state.vy = float(vf[1])

                PlayerStatesRepository.update_state(state, session)
                PlayerStatesRepository.update_state(prev_state, session)

                prev_state = state

        session.commit()


physics_procesor = PhysicsProcessing()
