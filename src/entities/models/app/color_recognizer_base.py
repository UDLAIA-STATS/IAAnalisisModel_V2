import traceback
from typing import List

import cv2
import logfire
import numpy as np
from skimage import color
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlmodel import Session

from src.core.repository.player_repository import PlayerRepository
from src.entities.models.soccer.player_model import PlayerModel
from src.entities.models.app.video_item import VideoItem

class ColorRecognizerBase:
    def _rgb_string_to_lab(self, color_str: str) -> np.ndarray:
        rgb = np.array(
            [int(x.strip()) for x in color_str.split(",")],
            dtype=np.float32,
        ) / 255.0

        return color.rgb2lab(rgb.reshape(1, 1, 3))[0, 0]

    def _lab_to_rgb_string(self, lab: np.ndarray) -> str:
        lab = np.asarray(lab, dtype=np.float32).reshape(1, 1, 3)

        rgb = color.lab2rgb(lab)[0, 0]
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        return f"{rgb[0]},{rgb[1]},{rgb[2]}"
    
    def validate_players(self, players: List[PlayerModel], session: Session) -> List[PlayerModel]:
        valid_players = []
        for player in players:
            if player.team_color is None:
                logfire.warning(
                    f"[ColorRecognizerBase] Deleting player {player.track_id} because team_color is None"
                )
                PlayerRepository.delete_player(player.id, session)
                continue

            valid_players.append(player)

        return valid_players

    def extract_color(self, crop: np.ndarray, n_clusters: int = 5, spatial_weight: float = 0.3):
        """
        Extracts jersey color using 5D Spatial-Color Clustering [L, a, b, x, y].
        Assumes 'crop' is already filtered to the player's bounding box.
        """
        h, w, _ = crop.shape

        lab_img = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        y_coords, x_coords = np.indices((h, w))

        colors_flat = lab_img.reshape((-1, 3))
        x_flat = x_coords.flatten().reshape(-1, 1)
        y_flat = y_coords.flatten().reshape(-1, 1)

        features = np.hstack((colors_flat, x_flat, y_flat)).astype(float)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        features_scaled[:, 3:] *= spatial_weight

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = kmeans.fit_predict(features_scaled)

        best_cluster_idx = -1
        best_score = float("inf")
        cluster_colors = []

        for i in range(n_clusters):
            cluster_mask = labels == i

            mean_y = np.mean(y_flat[cluster_mask])
            mean_x = np.mean(x_flat[cluster_mask])

            median_lab = np.median(colors_flat[cluster_mask], axis=0)
            cluster_colors.append(median_lab)

            center_penalty = abs(mean_x - (w / 2))

            score = mean_y + (center_penalty * 0.5)

            if score < best_score:
                best_score = score
                best_cluster_idx = i

        best_lab_color = np.uint8([[cluster_colors[best_cluster_idx]]])  # type: ignore
        best_rgb_color = cv2.cvtColor(best_lab_color, cv2.COLOR_LAB2RGB)[0][0]  # type: ignore

        rgb_tuple = tuple(map(int, best_rgb_color))
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

        return rgb_tuple, hex_color
    
    def color_distance(self, color_a: str, color_b: str) -> float:
        try:
            rgb_a = np.array(
                [int(x.strip()) for x in color_a.split(",")],
                dtype=np.float32,
            ) / 255.0

            rgb_b = np.array(
                [int(x.strip()) for x in color_b.split(",")],
                dtype=np.float32,
            ) / 255.0

            lab_a = color.rgb2lab(rgb_a.reshape(1, 1, 3))
            lab_b = color.rgb2lab(rgb_b.reshape(1, 1, 3))

            return float(color.deltaE_ciede2000(lab_a, lab_b)[0, 0])

        except Exception:
            logfire.error(f"Failed to calculate color distance {traceback.format_exc()}")
            raise

    def merge_colors(self, match_id: int, session: Session):
        pass

    def recognize(self, video_item: VideoItem, session: Session) -> List[str]:
        return []