import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ColorRecognizer:
    @staticmethod
    def extract_color(crop: np.ndarray, n_clusters: int = 3, spatial_weight: float = 0.8):
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

            # Total score (lower is better)
            score = mean_y + (center_penalty * 0.5)

            if score < best_score:
                best_score = score
                best_cluster_idx = i

        # 7. Convert the winning LAB color back to RGB
        best_lab_color = np.uint8([[cluster_colors[best_cluster_idx]]])  # type: ignore
        best_rgb_color = cv2.cvtColor(best_lab_color, cv2.COLOR_LAB2RGB)[0][0]  # type: ignore

        rgb_tuple = tuple(map(int, best_rgb_color))
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

        return rgb_tuple, hex_color

jersey_color_extractor = ColorRecognizer()