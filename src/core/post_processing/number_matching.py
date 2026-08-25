from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import logfire
from sqlmodel import Session, col, select

from src.entities.models.soccer.player_model import PlayerNumbers, PlayerState
from src.entities.models.number_recognizer.number_models import BBox, NumberObservation
from src.core.repository import PlayerRepository
from src.entities.models.pyspark.operations import bbox_iou


class NumberMatching:
    NONE_TENS = 10

    def __init__(
        self,
        iou_threshold: float = 0.15,
        frames_per_bucket: int = 30,
        illegible_threshold: float = 0.5,
        tens_none_confidence: float = 0.65,
    ):
        self.iou_threshold = iou_threshold
        self.frames_per_bucket = frames_per_bucket
        self.illegible_threshold = illegible_threshold
        self.tens_none_confidence = tens_none_confidence

    def iou(self, a: BBox, b: BBox) -> float:
        return bbox_iou(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2)

    def compute_occlusion_by_frame(
        self, bboxes: list[BBox]
    ) -> dict[tuple[int, int], float]:
        """Para cada (player_id, frame_number), devuelve el IoU máximo contra
        cualquier OTRO jugador en ese frame exacto. Agrupar por frame_number
        primero evita comparar contra jugadores de frames distintos."""
        by_frame: dict[int, list[BBox]] = defaultdict(list)
        for b in bboxes:
            by_frame[b.frame_number].append(b)

        max_iou: dict[tuple[int, int], float] = {}
        for boxes in by_frame.values():
            for i, bi in enumerate(boxes):
                best = 0.0
                for j, bj in enumerate(boxes):
                    if i != j:
                        best = max(best, self.iou(bi, bj))
                max_iou[(bi.player_id, bi.frame_number)] = best
        return max_iou

    def bucket_by_time(
        self, observations: list[NumberObservation], frames_per_bucket: int
    ) -> dict[int, list[NumberObservation]]:
        """Agrupa observaciones en ventanas temporales, para que una ráfaga de
        frames muestreados muy seguido cuente como ~un voto, no N votos."""
        buckets: dict[int, list[NumberObservation]] = defaultdict(list)
        for obs in observations:
            buckets[obs.frame_number // frames_per_bucket].append(obs)
        return buckets

    def consolidate_player(
        self,
        observations: list[NumberObservation],
        occlusion: dict[tuple[int, int], float],
    ) -> Optional[int]:
        """Voto por mayoría ponderado por confianza sobre las observaciones de
        UN jugador, extendido con filtrado de oclusión, des-duplicación de
        ráfagas y corrección de sesgo hacia dos dígitos."""
        clean = [
            obs
            for obs in observations
            if occlusion.get((obs.player_id, obs.frame_number), 0.0)
            < self.iou_threshold
        ]
        if not clean:
            return None  # nada confiable sobrevivió al filtrado

        buckets = self.bucket_by_time(clean, self.frames_per_bucket)
        representatives = [
            max(group, key=lambda o: o.confidence) for group in buckets.values()
        ]

        votes: dict[int, float] = defaultdict(float)
        total_weight = 0.0
        legible_weight = 0.0
        for obs in representatives:
            total_weight += obs.confidence
            if obs.number is None:
                continue
            weight = obs.confidence
            if (
                obs.tens_pred == self.NONE_TENS
                and obs.tens_none_prob < self.tens_none_confidence
            ):
                weight *= 0.5
            votes[obs.number] += weight
            legible_weight += obs.confidence

        if (
            total_weight == 0
            or legible_weight / total_weight < self.illegible_threshold
        ):
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0] if votes else None

    def process(self, match_id: int, session: Session) -> dict[int, Optional[int]]:
        players = PlayerRepository.get_players_by_match_id(match_id, session)
        player_ids = [p.id for p in players]
        if not player_ids:
            return {}

        number_rows = session.exec(
            select(PlayerNumbers).where(
                col(PlayerNumbers.player_id).in_(player_ids),
                col(PlayerNumbers.confidence) > 0.6,
            )
        ).all()

        if len(number_rows) == 0:
            number_rows = session.exec(
                select(PlayerNumbers).where(
                    col(PlayerNumbers.player_id).in_(player_ids),
                    col(PlayerNumbers.confidence) > 0.5,
                )
            ).all()

        state_rows = session.exec(
            select(PlayerState).where(col(PlayerState.player_id).in_(player_ids))
        ).all()

        logfire.info(
            f"[NumberMatcher] Loaded {len(number_rows)} number observations and {len(state_rows)} state observations"
        )

        observations_by_player: dict[int, list[NumberObservation]] = defaultdict(list)
        for r in number_rows:
            observations_by_player[r.player_id].append(
                NumberObservation(
                    player_id=r.player_id,
                    frame_number=r.frame_number,
                    number=r.number,
                    confidence=r.confidence,
                    visible_prob=r.visible_prob,
                    tens_pred=r.tens_pred,
                    tens_prob=r.tens_prob,
                    tens_none_prob=r.tens_none_prob,
                    units_pred=r.units_pred,
                    units_prob=r.units_prob,
                )
            )

        bboxes = [
            BBox(
                player_id=s.player_id,
                frame_number=s.frame_number,
                x1=s.x1,
                y1=s.y1,
                x2=s.x2,
                y2=s.y2,
            )
            for s in state_rows
            if s.x1 is not None
        ]
        occlusion = self.compute_occlusion_by_frame(bboxes)

        results: dict[int, Optional[int]] = {}
        for player in players:
            final_number = self.consolidate_player(
                observations_by_player.get(player.id, []), occlusion
            )

            if final_number is None:
                continue

            player.shirt_number = final_number
            results[player.id] = final_number

        session.add_all(players)
        session.commit()
        return results


number_postprocessing = NumberMatching()
