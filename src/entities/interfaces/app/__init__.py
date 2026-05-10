from abc import ABC, abstractmethod


class AnalysisStepHandler(ABC):
    number_step: int
    name: str
    retryable: bool =  True
    max_retries: int = 3

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, video_id: str, match_id: int, context: dict) -> dict:
        """
        Execute the step and return the results.
        param video_id: the video id
        param match_id: the match id
        param context: the context
        """
        pass
