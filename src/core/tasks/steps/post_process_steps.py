import logfire
from sqlmodel import Session

from src.core.repository.task_repository import TaskRepository
from src.core.services.player_validator import PlayerValidator
from src.entities.models.requests.queue_model import TaskStep
from src.core.post_processing.physics_processing import physics_procesor
from src.entities.types.states import StatesModel
from src.core.services.player_validator_physics import physical_validator
from src.core.repository.player_states_repository import PlayerStatesRepository

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

            PlayerValidator().validate(request.match_id, total_frames, session)
            physics_procesor.process(request.match_id, fps, session)
            needs_restart = physical_validator.validate(request.match_id, session)

            if needs_restart:
                PlayerStatesRepository.recalculate_physics(request.match_id, session)                
                physics_procesor.process(request.match_id, fps, session)

            validate_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(validate_step, session)
        except Exception as e:
            validate_step.state = StatesModel.FAILED
            validate_step.message = f"Error validando jugadores: {str(e)}"
            TaskRepository.upsert_task_step(validate_step, session)
            logfire.error(f"Error validating players: {traceback.format_exc()}")
            raise e