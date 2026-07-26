from collections import defaultdict
from typing import Dict, List, Set, override

import logfire
import numpy as np
from skimage import color
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlmodel import Session

from src.entities.models.app.color_recognizer_base import ColorRecognizerBase
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.core.utils.video_utils import extract_player_torso
from src.entities.models.app.video_item import VideoItem
from src.core.repository.player_repository import PlayerRepository

class ColorRecognizer(ColorRecognizerBase):
    @override
    def merge_colors(self, match_id: int, session: Session):
        players = PlayerRepository.get_players_by_match_id(match_id, session)

        if not players:
            # logfire.warning(f"[ColorRecognizer] No players found for match {match_id}")
            return

        valid_players = self.validate_players(list(players), session)

        if len(valid_players) < 2:
            logfire.info("[ColorRecognizer] Not enough players to merge colors.")
            return

        logfire.info(
            f"[ColorRecognizer] Building color graph with {len(valid_players)} players"
        )

        player_lab: Dict[int, np.ndarray] = {}

        for player in valid_players:
            player_lab[player.id] = self._rgb_string_to_lab(player.team_color)

        graph: Dict[int, Set[int]] = defaultdict(set)

        edge_count = 0

        for i in range(len(valid_players)):
            player_a = valid_players[i]

            for j in range(i + 1, len(valid_players)):
                player_b = valid_players[j]

                dist = float(
                    color.deltaE_ciede2000(
                        player_lab[player_a.id].reshape(1, 1, 3),
                        player_lab[player_b.id].reshape(1, 1, 3),
                    )[0, 0]
                )

                logfire.debug(
                    f"ΔE({player_a.track_id}, {player_b.track_id}) = {dist:.2f}"
                )

                if dist < 17.0:
                    graph[player_a.id].add(player_b.id)
                    graph[player_b.id].add(player_a.id)
                    edge_count += 1

        logfire.info(
            f"Graph built with {edge_count} similarity edges"
        )

        visited = set()
        components: List[List[int]] = []

        for player in valid_players:

            if player.id in visited:
                continue

            component = []
            stack = [player.id]

            while stack:
                node = stack.pop()

                if node in visited:
                    continue

                visited.add(node)
                component.append(node)

                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            components.append(component)

        logfire.info(
            f"[ColorRecognizer] Detected {len(components)} color groups"
        )

        player_lookup = {
            player.id: player
            for player in valid_players
        }

        for component_index, component in enumerate(components, start=1):
            labs = np.array(
                [player_lab[player_id] for player_id in component]
            )

            representative_lab = np.median(labs, axis=0)
            normalized_color = self._lab_to_rgb_string(representative_lab)

            track_ids = [
                player_lookup[player_id].track_id
                for player_id in component
            ]

            # logfire.info(
            #     f"[ColorRecognizer] Component {component_index}: "
            #     f"{len(component)} players | "
            #     f"Tracks={track_ids} | "
            #     f"Normalized color={normalized_color}"
            # )

            for player_id in component:
                player = player_lookup[player_id]

                if player.team_color != normalized_color:
                    logfire.debug(
                        f"[ColorRecognizer] Updating track {player.track_id}: "
                        f"{player.team_color} -> {normalized_color}"
                    )

                    player.team_color = normalized_color
                    PlayerRepository.upsert_player(player, session)

        logfire.info("Color normalization finished successfully.")
    # def merge_colors(self, match_id: int, session: Session):
    #     players = PlayerRepository.get_players_by_match_id(match_id, session)

    #     if players is None:
    #         return

    #     # lab_values: List[Tuple[PlayerModel, float]] = []
    #     used = set()

    #     for i in range(len(players)):
    #         if players[i].team_color is None:
    #             PlayerRepository.delete_player(players[i].id, session)
    #             used.add(players[i].id)
    #             continue

    #         for j in range(i + 1, len(players)):
    #             if players[j].id in used:
    #                 continue

    #             if players[j].team_color is None:
    #                 PlayerRepository.delete_player(players[j].id, session)
    #                 used.add(players[j].id)
    #                 continue

    #             dist = self.color_distance(players[i].team_color, players[j].team_color)
    #             logfire.info(f"Distance between {players[i].track_id} and {players[j].track_id}: {dist}")

    #             if dist < 18.0:
    #                 players[j].team_color = players[i].team_color
    #                 PlayerRepository.upsert_player(players[j], session)
    #                 used.add(players[j].id)

    @override
    def recognize(self, video_item: VideoItem, session: Session) -> List[str]:
        labels = []
        states = PlayerStatesRepository.get_states_by_frame(video_item.match_id, video_item.frame_num, session=session)

        if len(states) == 0:
            return labels

        crop = video_item.frame.copy()

        for state in states:
            if state.player.team_color is not None:
                hex = state.player.team_color
                label = f"ID: {state.player.track_id} |{hex}| Conf: {state.confidence:.2f}"
                labels.append(label)
                continue

            x1, y1, x2, y2 = state.x1, state.y1, state.x2, state.y2

            if x1 is None or y1 is None or x2 is None or y2 is None:
                # logfire.error(
                #     f"[ColorRecognition] No coordinates or coordinates incompleted for "
                #     f"player {state.player.track_id} in frame {video_item.frame_num} in match {video_item.match_id}"
                # )
                continue

            crop = extract_player_torso(crop, np.array([x1, y1, x2, y2]))

            if crop.shape[0] == 0 or crop.shape[1] == 0:
                # logfire.error(
                #     f"[ColorRecognition] No coordinates or coordinates incompleted for "
                #     f"player {state.player.track_id} in frame {video_item.frame_num} in match {video_item.match_id}"
                # )
                continue

            rgb, hex = self.extract_color(crop)
            rgb_str = f"{rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}"

            player = PlayerRepository.get_player_by_id(state.player_id, session)
            
            if player is None:
                logfire.error(f"[ColorRecognition] No player found for {state.player_id}")
                continue

            player.team_color = rgb_str

            label = f"ID: {state.player.track_id} |{hex}| Conf: {state.confidence:.2f}"
            labels.append(label)

            PlayerRepository.upsert_player(player, session)

        return labels



jersey_color_extractor = ColorRecognizer()
