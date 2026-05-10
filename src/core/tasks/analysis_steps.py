from entities.interfaces.app import AnalysisStepHandler


class ObjectDetection(AnalysisStepHandler):
    name = "Object Detection"
    number_step = 1

    def execute(self, video_id: str, match_id: int, context: dict) -> dict:
        return {}

class PhysicsCalculator(AnalysisStepHandler):
    name = "Physics Calculator"
    number_step = 2

    def execute(self, video_id: str, match_id: int, context: dict) -> dict:
        return {}
