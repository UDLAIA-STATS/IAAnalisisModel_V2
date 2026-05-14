from typing import Optional

from entities.interfaces.app import AnalysisStepHandler


class StepExecutor:
    def __init__(self):
        self.steps: list[AnalysisStepHandler] = []

    def get_step_by_number(self, step_number: int) -> Optional[AnalysisStepHandler]:
        return next((s for s in self.steps if s.number_step == step_number), None)

    def run_step(self, task_id: str, step_number: int):
        pass
