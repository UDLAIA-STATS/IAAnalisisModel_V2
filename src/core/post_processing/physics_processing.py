from sqlmodel import Session

from entities.services.physics_processing_model import PhysicsCalculatorBase


class PhysicsProcessing(PhysicsCalculatorBase):
    def process(self, match_id: int, constant: float, session: Session):
        states = self.get_players(match_id, session)

        if not states:
            return