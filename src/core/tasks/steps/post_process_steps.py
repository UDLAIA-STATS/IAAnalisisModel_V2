import logfire
from sqlmodel import Session

from src.core.repository.player_repository import PlayerRepository
from src.core.repository.task_repository import TaskRepository
from src.core.services.player_validator_spark import player_validator_cls
from src.entities.models.requests.queue_model import TaskStep
from src.core.post_processing.physics_processing import physics_procesor
from src.entities.types.states import StatesModel
from src.core.services.player_validator_physics_spark import physical_validator
from src.core.post_processing import (
    ball_predictor_cls,
    goal_validator_cls,
    number_postprocessing,
    ball_possession_analyzer,
    goal_scorer_detector_cls,
    player_crop_validator
)
from src.core.vision.color_recognizer import jersey_color_extractor

import traceback


class ValidationProcess:
    name = "Post Process Validation"
    number_step = 5

    def execute(self, session: Session, **kwargs):
        task = kwargs["task"]
        request = kwargs["request"]
        total_frames = kwargs["total_frames"]
        fps = kwargs["fps"]

        validate_step = TaskStep(
            task_id=task.id,
            name="Validación y Calculo de Medidas Físicas",
            message="Validando Jugadores y calculando medidas físicas",
            step_number=3,
        )
        TaskRepository.upsert_task_step(validate_step, session)

        try:

            player_validator_cls.validate(request.match_id, total_frames, session)
            physics_procesor.process(request.match_id, fps, session)
            physical_validator.validate(request.match_id, session)
            player_validator_cls.validate(request.match_id, total_frames, session)
            physics_procesor.process(request.match_id, fps, session)

            players = PlayerRepository.get_players_by_match_id(
                request.match_id, session
            )

            for idx, player in enumerate(players):
                player.track_id = idx + 1
                PlayerRepository.upsert_player(player, session)

            jersey_color_extractor.merge_colors(request.match_id, session)

            ball_predictor_cls.predict(request.match_id, total_frames, session)
            goal_validator_cls.validate(request.match_id, session)
            goal_scorer_detector_cls.detect(request.match_id, total_frames, session)

            number_postprocessing.process(request.match_id, session)
            ball_possession_analyzer.analyze(request.match_id, session)

            # player_crop_validator.validate_match(request.match_id, session)

            validate_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(validate_step, session)
        except Exception as e:
            validate_step.state = StatesModel.FAILED
            validate_step.message = f"Error validando jugadores: {str(e)}"
            TaskRepository.upsert_task_step(validate_step, session)
            logfire.error(f"Error validating players: {traceback.format_exc()}")
            raise e
