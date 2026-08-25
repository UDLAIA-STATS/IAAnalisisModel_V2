import logfire
from sqlmodel import Session

from src.core.video.video_manager import VideoManager
from src.core.repository.depth_history_repository import DepthRepository
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app.video_item import VideoItem
from src.entities.interfaces.app.analysis_step_handler import AnalysisStepHandler
from src.core.vision import (
    scale_motion_detector,
    player_depth_calculator,
    pixel_conversion_handler,
    pitch_homography,
    tilt_detector
)

from src.entities.models.soccer import DepthHistory


class ConversionCalculatorSteps(AnalysisStepHandler):
    name = "Constant Conversion Calculator"
    number_step = 4
    last_frame_calculated = 0
    last_kp_calculated = 0
    frame_step = 30

    def execute(self, session: Session, **kwargs) -> bool:
        video_item: VideoItem = kwargs["video_item"]
        manager: VideoManager = kwargs["video_manager"]

        states = PlayerStatesRepository.get_states_by_frame(
            video_item.match_id, video_item.frame_num, session=session
        )
        actual_scale = scale_motion_detector.get_current_scale()
        actual_pixel_conversion = pixel_conversion_handler.calculate_value(video_item.frame)
        actual_tilt = tilt_detector.get_current_tilt()
        
        if self.last_kp_calculated == 0 or self.last_kp_calculated + self.frame_step * 1.5 <= video_item.frame_num:
            result = pitch_homography.calibrate(
            video_item,
            actual_scale,
            actual_tilt,
            session,
        )

            video_item.annotated_frame = pitch_homography.draw_debug(
                video_item.annotated_frame, result
            )
            manager.write(video_item.annotated_frame, video_item.frame_num, save_frame=True)
            self.last_kp_calculated = video_item.frame_num

        try:
            for state in states:
                depth_history = DepthHistory(
                    player_id=state.player_id,
                    match_id=video_item.match_id,
                    frame_num=video_item.frame_num,
                    timestamp=video_item.timestamp,
                )

                if (
                    self.last_frame_calculated != video_item.frame_num
                    and video_item.frame_num - self.last_frame_calculated
                    > self.frame_step
                ):
                    actual_scale = scale_motion_detector.update(video_item.frame)
                    actual_tilt  = tilt_detector.update(video_item.frame)

                depth_history.pixels_to_meters = float(actual_pixel_conversion)
                depth_history.camera_scale = float(actual_scale)

                session.add(depth_history)
                session.flush()

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
