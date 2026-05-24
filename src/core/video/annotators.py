from typing import override

from supervision import Detections

from src.entities.services.annotator_service import AnnotatorServiceBase


class PlayerAnnotator(AnnotatorServiceBase):
    def __init__(
        self, anotator_name: str = "Player", hex_color: str = "#420D96", thickness: int = 2, text_thickness: int = 1, text_scale: float = 0.4
    ):
        super().__init__(anotator_name, hex_color, thickness, text_thickness, text_scale)

    @override
    def set_detections(self, detections: Detections):
        return super().set_detections(detections)


class BallAnnotator(AnnotatorServiceBase):
    def __init__(self, anotator_name: str = "Ball", hex_color: str = "#861818", thickness: int = 2, text_thickness: int = 1, text_scale: float = 0.5):
        super().__init__(anotator_name, hex_color, thickness, text_thickness, text_scale)

    @override
    def set_detections(self, detections: Detections):
        return super().set_detections(detections)


class GoalAnnotator(AnnotatorServiceBase):
    def __init__(self, anotator_name: str = "Goal", hex_color: str = "#CF9260", thickness: int = 2, text_thickness: int = 1, text_scale: float = 0.5):
        super().__init__(anotator_name, hex_color, thickness, text_thickness, text_scale)

    @override
    def set_detections(self, detections: Detections):
        return super().set_detections(detections)


player_annotator = PlayerAnnotator()
ball_annotator = BallAnnotator()
goal_annotator = GoalAnnotator()
