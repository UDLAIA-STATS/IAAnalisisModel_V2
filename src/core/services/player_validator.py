from typing import Dict, List, Tuple, Set
import logfire
from collections import defaultdict

from sqlmodel import Session

from src.entities.services.player_validator_base import PlayerValidatorBase, UnionFind
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.soccer.player_model import PlayerState

class PlayerValidator(PlayerValidatorBase):
    def validate(self, match_id: int, total_frames: int, session: Session) -> None:
        logfire.info(f"[PlayerValidator] Starting validation for match {match_id}")

        states = PlayerStatesRepository.get_states_by_frame_range(
            match_id=match_id,
            min_frame=1,
            max_frame=total_frames,
            session=session,
        )

        correct_states: Dict[int, List[PlayerState]] = defaultdict(list)
        incorrect_states: Dict[int, List[PlayerState]] = defaultdict(list)

        for state in states:
            if state.player.track_id <= 22:
                correct_states[state.frame_number].append(state)
            else:
                incorrect_states[state.frame_number].append(state)

        if not correct_states or not incorrect_states:
            logfire.info("[PlayerValidator] No data to validate")
            return

        correct_centroids = self._build_centroids(correct_states)
        incorrect_centroids = self._build_centroids(incorrect_states)

        correct_summaries = self._build_track_summaries(correct_centroids)
        incorrect_summaries = self._build_track_summaries(incorrect_centroids)

        ghost_ids: Set[int] = {
            pid for pid, s in incorrect_summaries.items()
            if s.detection_count <= self.GHOST_TRACK_THRESHOLD
        }
        logfire.info(
            f"[PlayerValidator] Correct tracks: {len(correct_summaries)} | "
            f"Incorrect tracks: {len(incorrect_summaries)} | "
            f"Ghost tracks: {ghost_ids}"
        )

        # ── score every valid (incorrect, correct) candidate pair ─────────────
        # A pair is valid only when:
        #   1. No frame overlap exists (hard constraint — same player can't be
        #      in two places at once)
        #   2. Color distance is below MAX_COLOR_DISTANCE
        #   3. Spatial re-entry distance is below 2 × MAX_DISTANCE
        #
        # Ghosts are prepended so the sort naturally resolves them first.

        scored_candidates: List[Tuple[float, int, int]] = []  # (score, incorrect_id, correct_id)

        for incorrect_id, inc in incorrect_summaries.items():
            is_ghost = incorrect_id in ghost_ids

            for correct_id, cor in correct_summaries.items():

                # ── hard constraint: temporal non-overlap ─────────────────────
                if self._has_frame_overlap(cor, inc):
                    continue

                color_dist = self.calculate_color_distance(inc.avg_color, cor.avg_color)
                if color_dist >= self.MAX_COLOR_DISTANCE:
                    continue

                # ── spatial re-entry: compare end of earlier track to start
                #    of the later one, not arbitrary frame centroids ──────────
                if cor.frame_end < inc.frame_start:
                    early, late = cor, inc
                else:
                    early, late = inc, cor

                gap = late.frame_start - early.frame_end

                if gap > self.MAX_FRAME_GAP:
                    continue

                pos_dist = self.calculate_distance(
                    (early.last_cx, early.last_cy),
                    (late.first_cx, late.first_cy),
                )

                if pos_dist >= self.MAX_DISTANCE * 2:
                    continue

                score = self._composite_score(color_dist, pos_dist, gap)

                # Bias ghosts slightly toward the front of the sorted list so
                # they are consumed before longer tracks compete for the same
                # correct_id.
                if is_ghost:
                    score *= 0.9

                scored_candidates.append((score, incorrect_id, correct_id))
                logfire.info(
                    f"[PlayerValidator] Candidate {incorrect_id}↔{correct_id}: "
                    f"score={score:.2f}  color_dist={color_dist:.1f}  "
                    f"pos_dist={pos_dist:.1f}  gap={gap}"
                )

        # ── greedy merge: consume pairs in score order ─────────────────────
        # Each incorrect_id is merged at most once; multiple correct_ids that
        # compete for the same incorrect_id are resolved by score ordering.
        # Union-Find propagates transitive chains automatically.

        scored_candidates.sort(key=lambda x: x[0])

        uf = UnionFind()
        merged_incorrect: Set[int] = set()

        for score, incorrect_id, correct_id in scored_candidates:
            if score > self.MERGE_SCORE_THRESHOLD:
                break
            if incorrect_id in merged_incorrect:
                continue

            uf.union(correct_id, incorrect_id)
            
            merged_incorrect.add(incorrect_id)
            logfire.info(
                f"[PlayerValidator] Scheduling merge {incorrect_id} → {correct_id} "
                f"(score={score:.2f})"
            )

        # ── execute merges derived from union-find groups ─────────────────
        # For each connected component, every non-root member is merged into
        # the root (canonical) ID.  This correctly handles chains such as
        # correct_id=3 ← incorrect_id=32 ← incorrect_id=38.

        merge_groups = uf.groups()
        total_merged = 0

        for root, members in merge_groups.items():
            for member in members:
                if member == root:
                    continue
                PlayerStatesRepository.merge_states(root, member, session)
                total_merged += 1

        session.commit()
        logfire.info(
            f"[PlayerValidator] Validation completed. "
            f"Merged {total_merged} tracks across {len(merge_groups)} groups."
        )
