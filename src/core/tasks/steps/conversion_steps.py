from sqlmodel import Session

from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app.video_item import VideoItem
from src.entities.interfaces.app.analysis_step_handler import AnalysisStepHandler
from src.core.vision import scale_motion_detector, player_depth_calculator, pixel_conversion_handler

from src.entities.models.soccer.depth_history import DepthHistory

class ConversionCalculatorSteps(AnalysisStepHandler):
    name = "Constant Conversion Calculator"
    number_step = 3

    def execute(self, session: Session, **kwargs) -> bool:
        video_item: VideoItem = kwargs["video_item"]
        default_constant = (
            scale_motion_detector.get_current_scale() * player_depth_calculator.get_last_depth() *
            pixel_conversion_handler.get_current_conversion())
        
        states = PlayerStatesRepository.get_states_by_frame(video_item.match_id, video_item.frame_num, session=session)
        actual_depth = player_depth_calculator.get_last_depth()
        actual_scale = scale_motion_detector.get_current_scale()
        actual_pixel_conversion = pixel_conversion_handler.get_current_conversion()
        constant = actual_depth * actual_scale * actual_pixel_conversion

        try:
            for state in states:
                depth_history = DepthHistory(
                    player_id=state.player_id,
                    match_id=video_item.match_id,
                    frame_num=video_item.frame_num,
                    timestamp=int(video_item.timestamp),
                )

                if video_item.frame_num % 30 != 0:
                    bbox = [int(state.x1), int(state.y1), int(state.x2), int(state.y2)]
                    actual_scale = scale_motion_detector.update(video_item.frame)
                    actual_depth = player_depth_calculator.process_player_depth(
                        bbox=bbox,
                        current_camera_scale=actual_scale,
                        frame=video_item.frame,
                        frame_num=video_item.frame_num,
                    )
                    actual_pixel_conversion = pixel_conversion_handler.calculate_value(video_item.frame)
                    constant = actual_depth * actual_scale * actual_pixel_conversion
                
                depth_history.depth = actual_depth
                depth_history.pixels_to_meters = actual_pixel_conversion
                depth_history.camera_scale = actual_scale 
                depth_history.constant = constant

                session.add(depth_history)
                session.flush()

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e