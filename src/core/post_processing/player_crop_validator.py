from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logfire
import numpy as np
import cv2
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import torch
from ultralytics import YOLO
from sqlmodel import Session, select

from src.core.repository.player_repository import PlayerRepository
from src.core.repository.r2_repository import R2Repository as r2_repo
from src.entities.models.soccer.player_model import (
    PlayerModel,
    PlayerState,
    PlayerNumbers,
    DepthHistory,
)

from src.config.routes import DEPTH_MODEL_PATH, INPUT_VIDEOS_DIR, PLAYER_MODEL_PATH
from src.config.configuration import settings
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PlayerCropValidator:
    """
    Valida y unifica jugadores a partir de sus imágenes recortadas (crop_path).
    """

    YOLO_CONFIDENCE_THRESHOLD: float = 0.15
    COSINE_DISTANCE_THRESHOLD: float = 0.1
    MIN_PLAYERS_TO_CLUSTER: int = 8

    YOLO_MODEL_PATH: str = PLAYER_MODEL_PATH.as_posix()

    def __init__(self, reid_model_name: str = "osnet_x1_0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logfire.info(f"[PlayerCropValidator] Using device: {self.device}")

        self.yolo = YOLO(self.YOLO_MODEL_PATH)
        self.reid_model = self._load_reid_model(reid_model_name)

        self.pose_landmarker = None

        base_options = python.BaseOptions(model_asset_path=DEPTH_MODEL_PATH.as_posix())
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    def _estimate_blur(self, image: np.ndarray) -> float:
        """Calcula la varianza del Laplaciano para medir la nitidez."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _detect_with_pose(self, image: np.ndarray) -> Tuple[List, float]:
        """
        Detecta puntos clave (keypoints) usando MediaPipe Tasks API (PoseLandmarker).
        Retorna: (lista de bboxes, confianza)
        """
        if self.pose_landmarker is None:
            return [], 0.0

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            result = self.pose_landmarker.detect(mp_image)

            if not result.pose_landmarks:
                return [], 0.0

            landmarks = result.pose_landmarks[0]
            
            h, w, _ = image.shape
            xs, ys = [], []

            for landmark in landmarks:
                xs.append(landmark.x * w)
                ys.append(landmark.y * h)

            if xs and ys:
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                margin_x = (x2 - x1) * 0.15
                margin_y = (y2 - y1) * 0.15
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)

                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    logfire.debug("[PlayerCropValidator] Pose detectada exitosamente mediante MediaPipe Tasks.")
                    return [(x1, y1, x2, y2, 0.7)], 0.7

        except Exception as e:
            logfire.warning(f"[PlayerCropValidator] Error en la detección de pose con Tasks API: {e}")

        return [], 0.0

    def _load_reid_model(self, model_name: str):
        """Carga modelo pre-entrenado de torchreid."""
        try:
            from torchreid import models

            model = models.build_model(
                name=model_name, num_classes=1000, pretrained=True
            )
            model = model.model

            if hasattr(model, "classifier"):
                model.classifier = torch.nn.Identity()
            model.eval().to(self.device)
            logfire.info(f"[PlayerCropValidator] Loaded ReID model: {model_name}")
            return model
        except Exception as e:
            logfire.error(f"[PlayerCropValidator] Failed to load ReID model: {e}")

            from torchvision import models as tv_models

            model = tv_models.resnet50(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval().to(self.device)
            logfire.warning(
                "[PlayerCropValidator] Using ResNet50 as fallback ReID model"
            )
            return model

    def validate_match(self, match_id: int, session: Session) -> None:
        logfire.info(f"[PlayerCropValidator] Starting validation for match {match_id}")

        players = PlayerRepository.get_players_by_match_id(match_id, session)
        if not players:
            logfire.info("[PlayerCropValidator] No players found for match")
            return

        logfire.info(f"[PlayerCropValidator] Loaded {len(players)} players")

        player_data = []
        to_delete_ids = []

        for player in players:
            logfire.info(
                f"[PlayerCropValidator] Processing player {player.id} (track {player.track_id})"
            )

            if not player.crop_path:
                logfire.warning(
                    f"[PlayerCropValidator] Player {player.id} has no crop_path, marking for deletion"
                )
                to_delete_ids.append(player.id)
                continue

            try:
                image = self._download_image(player.crop_path)
                if image is None:
                    logfire.warning(
                        f"[PlayerCropValidator] Failed to download image for player {player.id}"
                    )
                    to_delete_ids.append(player.id)
                    continue

                _, max_conf = self._detect_person(image)
                if max_conf < self.YOLO_CONFIDENCE_THRESHOLD:
                    logfire.info(
                        f"[PlayerCropValidator] Player {player.id}: no person detected (conf={max_conf:.2f}), marking for deletion"
                    )
                    to_delete_ids.append(player.id)
                    continue

                embedding = self._extract_embedding(image)
                player_data.append(
                    {
                        "player_id": player.id,
                        "embedding": embedding,
                        "track_id": player.track_id,
                        "conf": max_conf,
                    }
                )
                logfire.info(
                    f"[PlayerCropValidator] Player {player.id}: valid person detected, embedding extracted"
                )

            except Exception as e:
                logfire.error(
                    f"[PlayerCropValidator] Error processing player {player.id}: {e}"
                )
                to_delete_ids.append(player.id)

        if len(player_data) < self.MIN_PLAYERS_TO_CLUSTER:
            logfire.info(
                "[PlayerCropValidator] Not enough valid players to cluster, only deleting invalid ones"
            )
            self._delete_players(to_delete_ids, session)
            return

        clusters = self._cluster_embeddings(player_data)

        merge_map = {}
        primary_players = set()

        for cluster in clusters:
            if len(cluster) > 1:
                primary_id = self._select_primary_player(cluster, player_data)
                primary_players.add(primary_id)
                for other_id in cluster:
                    if other_id != primary_id:
                        merge_map[other_id] = primary_id
            else:
                primary_players.add(cluster[0])

        self._apply_merges_and_deletions(
            match_id, merge_map, to_delete_ids, primary_players, session
        )

        logfire.info(f"[PlayerCropValidator] Validation completed for match {match_id}")

    def _download_image(self, crop_url: str) -> Optional[np.ndarray]:
        """
        Descarga la imagen desde R2 usando el repositorio R2Repository.
        """
        try:
            base_url = settings.PLAYER_DATA_PUBLIC_URL
            if crop_url.startswith(base_url):
                key = crop_url.split(settings.PLAYER_DATA_PUBLIC_URL)[-1].strip()[1:]
                # 369/players/369_614_88_7e4a41f0-7066-438f-838c-9e78833ce3e2.png
                logfire.info(
                    f"[PlayerCropValidator] Downloading image from R2 cutted key: {key} from {crop_url}"
                )
            else:
                # Si no coincide, intentar usar la URL directamente (podría ser key)
                key = crop_url

            pathfile = Path(INPUT_VIDEOS_DIR) / key
            pathfile.parent.mkdir(parents=True, exist_ok=True)
            r2_repo().download_player_image(
                key=key, destination_path=pathfile.as_posix()
            )
            img = cv2.imread(pathfile.as_posix())
            pathfile.unlink(missing_ok=True)
            return img

        except Exception as e:
            logfire.error(
                f"[PlayerCropValidator] Error downloading image from {crop_url}: {e}"
            )
            return None

    def _detect_person(self, image: np.ndarray) -> Tuple[List, float]:
        """
        Usa YOLO para detectar personas en la imagen.
        Retorna: (lista de bboxes, confianza máxima)
        """
        blur_score = self._estimate_blur(image)
        enhanced_image = self._sharpen_image(image) if blur_score < 50.0 else image

        results = self.yolo.predict(
            enhanced_image,
            augment=True,
            conf=0.15,
            verbose=False,
            device=self.device,
            agnostic_nms=True,
        )

        max_conf = 0.0
        person_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # persona en COCO
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    person_boxes.append((x1, y1, x2, y2, conf))
        
        if max_conf < self.YOLO_CONFIDENCE_THRESHOLD:
            color_boxes, color_conf = self._detect_by_color(enhanced_image)
            if color_conf > 0.0:
                logfire.info("[PlayerCropValidator] Fallback detection succeeded via COLOR segmentation.")
                return color_boxes, color_conf

            pose_boxes, pose_conf = self._detect_with_pose(enhanced_image)
            if pose_conf > 0.0:
                logfire.info("[PlayerCropValidator] Fallback detection succeeded via POSE keypoints.")
                return pose_boxes, pose_conf

        return person_boxes, max_conf

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen y extrae embedding usando el modelo ReID.
        """
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.reid_model(tensor).cpu().numpy().flatten()
        # Normalizar
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def _cluster_embeddings(self, player_data: List[Dict]) -> List[List[int]]:
        """
        Agrupa embeddings similares usando distancia coseno y linkage jerárquico.
        Retorna lista de clusters (cada cluster es lista de player_ids).
        """
        if len(player_data) < 2:
            return [[d["player_id"] for d in player_data]]

        ids = [d["player_id"] for d in player_data]
        embeddings = np.array([d["embedding"] for d in player_data])

        # Matriz de distancia coseno
        dist_matrix = pdist(embeddings, metric="cosine")
        # Linkage jerárquico (average linkage)
        Z = linkage(dist_matrix, method="average")
        # Cortar por umbral de distancia
        clusters = fcluster(Z, self.COSINE_DISTANCE_THRESHOLD, criterion="distance")
        # Agrupar IDs por cluster
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            cluster_dict.setdefault(cluster_id, []).append(ids[idx])
        return list(cluster_dict.values())

    def _select_primary_player(
        self, cluster_ids: List[int], player_data: List[Dict]
    ) -> int:
        """
        Elige el jugador principal del cluster (el que tiene mayor confianza de detección).
        """
        best = max(
            cluster_ids,
            key=lambda pid: next(
                d["conf"] for d in player_data if d["player_id"] == pid
            ),
        )
        return best

    def _apply_merges_and_deletions(
        self,
        match_id: int,
        merge_map: Dict[int, int],
        to_delete: List[int],
        primary_ids: set,
        session: Session,
    ) -> None:
        """
        Aplica fusiones y elimina jugadores inválidos.
        - Para cada secondary -> primary, mueve todas las relaciones (PlayerState, Numbers, DepthHistory).
        - Elimina secondary y los que están en to_delete (si no son primary).
        """
        stmt = select(PlayerModel).where(PlayerModel.match_id == match_id)
        all_players = session.exec(stmt).all()
        player_dict = {p.id: p for p in all_players}

        for secondary_id, primary_id in merge_map.items():
            if secondary_id not in player_dict or primary_id not in player_dict:
                logfire.warning(
                    f"[PlayerCropValidator] Skipping merge: player {secondary_id} or {primary_id} not found"
                )
                continue
            secondary = player_dict[secondary_id]
            primary = player_dict[primary_id]

            self._transfer_relations(secondary, primary, session)
            session.delete(secondary)
            logfire.info(
                f"[PlayerCropValidator] Merged player {secondary_id} into {primary_id}"
            )

        # 2. Eliminar jugadores inválidos (los que no tienen persona y no son primary)
        final_delete = [
            pid for pid in to_delete if pid not in primary_ids and pid in player_dict
        ]
        for pid in final_delete:
            player = player_dict.get(pid)
            if player:
                session.delete(player)
                logfire.info(f"[PlayerCropValidator] Deleted invalid player {pid}")

        session.commit()
        logfire.info(
            f"[PlayerCropValidator] Applied merges and deletions for match {match_id}"
        )

    def _transfer_relations(
        self, source: PlayerModel, target: PlayerModel, session: Session
    ):
        """
        Transfiere todas las relaciones (PlayerState, PlayerNumbers, DepthHistory) de source a target.
        """
        # PlayerState
        stmt = select(PlayerState).where(PlayerState.player_id == source.id)
        states = session.exec(stmt).all()
        for s in states:
            s.player_id = target.id
            session.add(s)

        # PlayerNumbers
        stmt = select(PlayerNumbers).where(PlayerNumbers.player_id == source.id)
        nums = session.exec(stmt).all()
        for n in nums:
            n.player_id = target.id
            session.add(n)

        # DepthHistory
        stmt = select(DepthHistory).where(DepthHistory.player_id == source.id)
        depths = session.exec(stmt).all()
        for d in depths:
            d.player_id = target.id
            session.add(d)

        # Nota: si hay otras relaciones, agregarlas aquí
        logfire.info(
            f"[PlayerCropValidator] Transferred relationships from player {source.id} to {target.id}"
        )

    def _delete_players(self, player_ids: List[int], session: Session) -> None:
        """Elimina jugadores de la base de datos (cascada automática si está configurada)."""
        for pid in player_ids:
            player = session.get(PlayerModel, pid)
            if player:
                session.delete(player)
        session.commit()
        logfire.info(f"[PlayerCropValidator] Deleted {len(player_ids)} players")

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Aplica un filtro de nitidez (Unsharp Mask)."""
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def _detect_by_color(self, image: np.ndarray) -> Tuple[List, float]:
        """
        Verifica la presencia de un jugador basándose en el color dominante (uniforme).
        Retorna: (lista de bboxes, confianza)
        """
        h, w, _ = image.shape
        center_roi = image[h//4:3*h//4, w//4:3*w//4]
        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        
        # Calcular histograma del tono (Hue) en la región central
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hist))
        
        # Crear máscara alrededor del color dominante (rango +/- 10)
        lower_color = np.array([max(0, dominant_hue - 10), 50, 50])
        upper_color = np.array([min(179, dominant_hue + 10), 255, 255])
        
        full_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(full_hsv, lower_color, upper_color)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_cnt) > (h * w * 0.05):  # Al menos 5% del área
                x, y, w_box, h_box = cv2.boundingRect(largest_cnt)
                return [(float(x), float(y), float(x + w_box), float(y + h_box), 0.6)], 0.6
        return [], 0.0
    



player_crop_validator = PlayerCropValidator()
