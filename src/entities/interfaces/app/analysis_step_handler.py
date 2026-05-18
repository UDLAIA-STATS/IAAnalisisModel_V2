from abc import ABC, abstractmethod
from typing import Dict

from sqlmodel import Session


class AnalysisStepHandler(ABC):
    number_step: int
    name: str
    retryable: bool = True
    max_retries: int = 3

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, session: Session, **kwargs) -> bool:
        """
        Execute the step and return the results.
        param video_id: the video id
        param match_id: the match id
        param context: the context
        """
        pass
