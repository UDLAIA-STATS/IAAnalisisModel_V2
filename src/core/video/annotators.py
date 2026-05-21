from src.entities.services.annotator_service import AnnotatorServiceBase
from src.entities.utils.singleton import Singleton


class PlayerAnnotator(AnnotatorServiceBase, metaclass=Singleton):
    def __init__(
        self, anotator_name: str = "Player", hex_color: str = "#420D96", thickness: int = 2, text_thickness: int = 1, text_scale: float = 0.5
    ):
        super().__init__(anotator_name, hex_color, thickness, text_thickness, text_scale)


class BallAnnotator(AnnotatorServiceBase, metaclass=Singleton):
    def __init__(self, anotator_name: str = "Ball", hex_color: str = "#861818", thickness: int = 2, text_thickness: int = 1, text_scale: float = 0.5):
        super().__init__(anotator_name, hex_color, thickness, text_thickness, text_scale)


class GoalAnnotator(AnnotatorServiceBase, metaclass=Singleton):
    def __init__(self, anotator_name: str = "Goal", hex_color: str = "#CF9260", thickness: int = 2, text_thickness: int = 1, text_scale: float = 0.5):
        super().__init__(anotator_name, hex_color, thickness, text_thickness, text_scale)


player_annotator = PlayerAnnotator()
ball_annotator = BallAnnotator()
goal_annotator = GoalAnnotator()
