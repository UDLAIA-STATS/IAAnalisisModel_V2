from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logfire
import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
from skimage.color import rgb2lab, deltaE_ciede2000
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
from src.config.routes import INPUT_VIDEOS_DIR
from src.config.configuration import settings
from src.core.vision.color_recognizer import jersey_color_extractor


class PlayerCropValidator:
    """
    Valida y unifica jugadores a partir de sus imágenes recortadas (crop_path).

    Pipeline de validación (por jugador):

      1) PRESENCIA / COMPLETITUD -> puede ELIMINAR el crop
         - Etapa primaria: detección de persona con YOLO (confianza >= 0.15).
         - Etapa de respaldo: segmentación por regiones (MSER). Si se encuentran más de
           MIN_SEGMENTS_FOR_COMPLETENESS regiones, se considera que el jugador está presente.
         - El crop SOLO se elimina si AMBAS etapas fallan.
         - Se extrae también el bounding box de la persona (para filtros de aspecto).

      2) EXTRACCIÓN DE ATRIBUTOS (para cada jugador válido)
         - Embedding ReID (OSNet)
         - Color de camiseta (RGB, LAB, HEX)
         - Número de camiseta (de la BD)
         - Bounding box de la persona (de YOLO)
         - Embedding facial (si se detecta rostro)
         - Keypoints de pose (si se detecta, con YOLO-pose)

      3) SIMILITUD / FUSIÓN (jerarquizada, nunca elimina, solo combina)
         - Nivel 1 (bloqueo duro): número de camiseta distinto -> NO fusionar.
         - Nivel 2 (bloqueo duro): color de camiseta incompatible (CIEDE2000 > umbral).
         - Nivel 3 (bloqueo duro): relación de aspecto o área incompatible.
         - Nivel 4 (bloqueo duro): si ambos tienen rostro, distancia facial > umbral.
         - Nivel 5 (bloqueo duro): si ambos tienen pose, proporciones corporales muy distintas.
         - Nivel 6 (criterio final): distancia coseno entre embeddings ReID.
           Si <= COSINE_DISTANCE_THRESHOLD -> mismo jugador y se fusionan.
    """

    # Umbrales de presencia / calidad
    YOLO_CONFIDENCE_THRESHOLD: float = 0.15
    MIN_SEGMENTS_FOR_COMPLETENESS: int = 3
    MIN_PLAYERS_TO_CLUSTER: int = 3

    COLOR_DISTANCE_THRESHOLD: float = 16.0      # CIEDE2000
    ASPECT_RATIO_TOLERANCE: float = 0.25
    AREA_TOLERANCE: float = 0.3
    FACE_DISTANCE_THRESHOLD: float = 0.2        # para face_recognition (coseno)
    POSE_RATIO_TOLERANCE: float = 0.15

    COSINE_DISTANCE_THRESHOLD: float = 0.12
    INCOMPATIBLE_NUMBER_DISTANCE: float = 999.0

    def __init__(self, reid_model_name: str = "osnet_x1_0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logfire.info(f"[PlayerCropValidator] Using device: {self.device}")

        self.yolo = YOLO("yolo26m.pt")                # Para detección de persona
        self.yolo_pose = YOLO("yolo26m-pose.pt")      # Para keypoints (opcional)
        self.reid_model = self._load_reid_model(reid_model_name)
        self.color_extractor = jersey_color_extractor

        self._face_encoder = None

    def _load_reid_model(self, model_name: str):
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
            logfire.warning("[PlayerCropValidator] Using ResNet50 as fallback ReID model")
            return model

    def _get_face_encoder(self):
        """Carga el modelo de face_recognition (solo si se necesita)."""
        if self._face_encoder is None:
            try:
                import face_recognition
                self._face_encoder = face_recognition
                logfire.info("[PlayerCropValidator] Loaded face_recognition")
            except ImportError:
                logfire.warning("[PlayerCropValidator] face_recognition not installed; facial filter disabled")
                self._face_encoder = False
        return self._face_encoder if self._face_encoder is not False else None

    def _estimate_blur(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def _detect_person_yolo(self, image: np.ndarray) -> Tuple[List[Tuple[float, float, float, float, float]], float]:
        """Etapa primaria: detección de persona con YOLO. Devuelve bboxes y conf máxima."""
        results = self.yolo.predict(
            image,
            conf=self.YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
            device=self.device,
            agnostic_nms=True,
        )

        person_boxes = []
        max_conf = 0.0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # persona
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    person_boxes.append((x1, y1, x2, y2, conf))
        return person_boxes, max_conf

    def _detect_segmentation_backup(self, image: np.ndarray) -> Tuple[List, float, bool]:
        """Etapa de respaldo: MSER. Devuelve (regiones, confianza_ficticia, ok)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER.create()
        regions, _ = mser.detectRegions(gray)
        min_area = 100
        valid_regions = [r for r in regions if r.shape[0] >= min_area]

        if len(valid_regions) > self.MIN_SEGMENTS_FOR_COMPLETENESS:
            h, w, _ = image.shape
            box = (0, 0, w, h, 0.5)
            logfire.info(
                f"[PlayerCropValidator] Segmentación de respaldo: {len(valid_regions)} regiones (> {self.MIN_SEGMENTS_FOR_COMPLETENESS})"
            )
            return [box], 0.5, True
        else:
            logfire.info(
                f"[PlayerCropValidator] Segmentación de respaldo: solo {len(valid_regions)} regiones, insuficiente"
            )
            return [], 0.0, False

    def _check_presence_and_completeness(self, image: np.ndarray) -> Tuple[bool, float, str, Optional[Tuple[float, float, float, float]]]:
        """
        Verifica presencia y completitud.
        Retorna (es_valido, confianza, metodo_usado, bbox_persona_mas_confiable).
        El bbox se usa para filtros de aspecto.
        """
        blur_score = self._estimate_blur(image)
        enhanced_image = self._sharpen_image(image) if blur_score < 50.0 else image

        person_boxes, yolo_conf = self._detect_person_yolo(enhanced_image)
        best_bbox = None
        if person_boxes:
            best_bbox = max(person_boxes, key=lambda x: x[4])[:4]  # (x1,y1,x2,y2)

        if yolo_conf >= self.YOLO_CONFIDENCE_THRESHOLD:
            return True, yolo_conf, "yolo", best_bbox

        _, seg_conf, seg_ok = self._detect_segmentation_backup(enhanced_image)
        if seg_ok:
            h, w, _ = image.shape
            fallback_bbox = (0, 0, w, h)
            return True, seg_conf, "segmentation_backup", fallback_bbox

        return False, 0.0, "none", None


    def _extract_color(self, image: np.ndarray) -> Tuple[Tuple[int, int, int], str]:
        rgb, hex_color = self.color_extractor.extract_color(image)
        return rgb, hex_color

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Optional[np.ndarray]:
        try:
            rgb_norm = np.array(rgb, dtype=np.float32) / 255.0
            lab = rgb2lab(rgb_norm.reshape(1, 1, 3)).reshape(3)
            return lab
        except Exception as e:
            logfire.warning(f"[PlayerCropValidator] Error converting RGB to LAB: {e}")
            return None

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extrae embedding ReID con conversión BGR->RGB."""
        from torchvision import transforms
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.reid_model(tensor).cpu().numpy().flatten()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def _extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Usa face_recognition para obtener embedding facial (128d). Retorna None si no se detecta rostro."""
        encoder = self._get_face_encoder()
        if encoder is None:
            return None
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = encoder.face_locations(rgb, model="hog")
            if not face_locations:
                return None
            encodings = encoder.face_encodings(rgb, face_locations)
            if encodings:
                return np.array(encodings[0])
        except Exception as e:
            logfire.warning(f"[PlayerCropValidator] Face embedding failed: {e}")
        return None

    def _extract_pose_keypoints(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extrae keypoints de pose con YOLO-pose y calcula proporciones corporales.
        Retorna dict con 'torso_leg_ratio' y 'shoulder_hip_ratio', o None si falla.
        """
        try:
            results = self.yolo_pose.predict(image, conf=0.3, verbose=False, device=self.device)
            if not results or results[0].keypoints is None:
                return None
            kps = results[0].keypoints.xy.cpu().numpy()[0]  # (17,2)
            if kps.shape[0] < 17:
                return None

            # Índices COCO: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip,
            # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
            def dist(a, b):
                return np.linalg.norm(kps[a] - kps[b]) if kps[a,0] > 0 and kps[b,0] > 0 else 0.0

            shoulder_center = (kps[5] + kps[6]) / 2
            hip_center = (kps[11] + kps[12]) / 2
            torso_len = np.linalg.norm(shoulder_center - hip_center)

            knee_center = (kps[13] + kps[14]) / 2
            ankle_center = (kps[15] + kps[16]) / 2
            leg_len = np.linalg.norm(knee_center - ankle_center)  # o desde hip a ankle

            shoulder_width = np.linalg.norm(kps[5] - kps[6])
            hip_width = np.linalg.norm(kps[11] - kps[12])

            if leg_len == 0 or hip_width == 0:
                return None

            ratios = {
                'torso_leg_ratio': float(torso_len / leg_len),
                'shoulder_hip_ratio': float(shoulder_width / hip_width)
            }
            return ratios
        except Exception as e:
            logfire.warning(f"[PlayerCropValidator] Pose extraction failed: {e}")
            return None

    # --------------------- Filtros de compatibilidad (bloqueantes) ---------------------

    def _color_compatible(self, lab1: Optional[np.ndarray], lab2: Optional[np.ndarray]) -> bool:
        if lab1 is None or lab2 is None:
            return True
        try:
            dist = deltaE_ciede2000(lab1, lab2)
            return dist <= self.COLOR_DISTANCE_THRESHOLD
        except:
            return True

    def _aspect_compatible(self, bbox1: Optional[Tuple], bbox2: Optional[Tuple]) -> bool:
        if bbox1 is None or bbox2 is None:
            return True
        x1,y1,x2,y2 = bbox1
        h1, w1 = y2-y1, x2-x1
        x1,y1,x2,y2 = bbox2
        h2, w2 = y2-y1, x2-x1
        if w1 == 0 or w2 == 0:
            return True
        ratio1 = h1 / w1
        ratio2 = h2 / w2
        area1 = h1 * w1
        area2 = h2 * w2
        if abs(ratio1 - ratio2) > self.ASPECT_RATIO_TOLERANCE:
            return False
        if abs(area1 - area2) / max(area1, area2) > self.AREA_TOLERANCE:
            return False
        return True

    def _face_compatible(self, emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> bool:
        if emb1 is None or emb2 is None:
            return True
        try:
            dist = cosine(emb1, emb2)
            return dist <= self.FACE_DISTANCE_THRESHOLD
        except:
            return True

    def _pose_compatible(self, pose1: Optional[Dict], pose2: Optional[Dict]) -> bool:
        if pose1 is None or pose2 is None:
            return True
        try:
            for key in ['torso_leg_ratio', 'shoulder_hip_ratio']:
                if abs(pose1[key] - pose2[key]) > self.POSE_RATIO_TOLERANCE:
                    return False
            return True
        except:
            return True

    # --------------------- Fusión jerarquizada (Condición 2) ---------------------

    def _pair_distance(self, i: int, j: int, player_data: List[Dict]) -> float:
        """
        Calcula distancia entre dos jugadores aplicando todos los filtros bloqueantes.
        Si alguno falla, retorna distancia infinita.
        """
        d1 = player_data[i]
        d2 = player_data[j]

        # 1. Número de camiseta (bloqueo duro)
        if d1["shirt_number"] is not None and d2["shirt_number"] is not None and d1["shirt_number"] != d2["shirt_number"]:
            return self.INCOMPATIBLE_NUMBER_DISTANCE

        # 2. Color
        if not self._color_compatible(d1.get("color_lab"), d2.get("color_lab")):
            return self.INCOMPATIBLE_NUMBER_DISTANCE

        # 3. Aspecto y área
        if not self._aspect_compatible(d1.get("person_bbox"), d2.get("person_bbox")):
            return self.INCOMPATIBLE_NUMBER_DISTANCE

        # 4. Facial
        if not self._face_compatible(d1.get("face_embedding"), d2.get("face_embedding")):
            return self.INCOMPATIBLE_NUMBER_DISTANCE

        # 5. Pose
        if not self._pose_compatible(d1.get("pose_keypoints"), d2.get("pose_keypoints")):
            return self.INCOMPATIBLE_NUMBER_DISTANCE

        # 6. Embedding ReID (criterio final)
        return cosine(d1["embedding"], d2["embedding"])

    def _build_distance_matrix(self, player_data: list) -> np.ndarray:
        n = len(player_data)
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(self._pair_distance(i, j, player_data))
        return np.array(condensed)

    def _cluster_by_identity(self, player_data: list) -> list:
        if len(player_data) < 2:
            return [[d["player_id"] for d in player_data]]

        ids = [d["player_id"] for d in player_data]
        condensed = self._build_distance_matrix(player_data)

        Z = linkage(condensed, method="average")
        cluster_labels = fcluster(Z, self.COSINE_DISTANCE_THRESHOLD, criterion="distance")

        cluster_dict = {}
        for idx, cluster_id in enumerate(cluster_labels):
            cluster_dict.setdefault(cluster_id, []).append(ids[idx])
        return list(cluster_dict.values())

    def _select_primary_player(self, cluster_ids: List[int], player_data: List[Dict]) -> int:
        return max(
            cluster_ids,
            key=lambda pid: next(d["conf"] for d in player_data if d["player_id"] == pid),
        )

    # --------------------- Método principal ---------------------

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
            logfire.info(f"[PlayerCropValidator] Processing player {player.id} (track {player.track_id})")

            if not player.crop_path:
                logfire.warning(f"[PlayerCropValidator] Player {player.id} has no crop_path, marking for deletion")
                to_delete_ids.append(player.id)
                continue

            try:
                image = self._download_image(player.crop_path)
                if image is None:
                    logfire.warning(f"[PlayerCropValidator] Failed to download image for player {player.id}")
                    to_delete_ids.append(player.id)
                    continue

                # Condición 1: presencia y completitud (solo elimina si ambas fallan)
                is_present, conf, method, person_bbox = self._check_presence_and_completeness(image)

                if not is_present:
                    logfire.info(f"[PlayerCropValidator] Player {player.id}: no se detectó jugador (YOLO y segmentación fallaron), marcando para eliminación")
                    to_delete_ids.append(player.id)
                    continue

                # Extraer todos los atributos
                embedding = self._extract_embedding(image)
                color_rgb, color_hex = self._extract_color(image)
                color_lab = self._rgb_to_lab(color_rgb)
                face_emb = self._extract_face_embedding(image)
                pose_kps = self._extract_pose_keypoints(image)

                player_data.append({
                    "player_id": player.id,
                    "embedding": embedding,
                    "track_id": player.track_id,
                    "conf": conf,
                    "color_rgb": color_rgb,
                    "color_lab": color_lab,
                    "color_hex": color_hex,
                    "shirt_number": player.shirt_number,
                    "person_bbox": person_bbox,          # (x1,y1,x2,y2) o None
                    "face_embedding": face_emb,
                    "pose_keypoints": pose_kps,
                })

                logfire.info(f"[PlayerCropValidator] Player {player.id}: válido (método={method}, conf={conf:.2f})")

            except Exception as e:
                logfire.error(f"[PlayerCropValidator] Error processing player {player.id}: {e}")
                to_delete_ids.append(player.id)

        # Si no hay suficientes jugadores válidos, solo eliminar los inválidos
        if len(player_data) < self.MIN_PLAYERS_TO_CLUSTER:
            logfire.info("[PlayerCropValidator] Not enough valid players to cluster, only deleting invalid ones")
            self._delete_players(to_delete_ids, session)
            return

        # Clustering y fusión (Condición 2)
        clusters = self._cluster_by_identity(player_data)

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

    # --------------------- Helpers de persistencia ---------------------

    def _download_image(self, crop_url: str) -> Optional[np.ndarray]:
        try:
            base_url = settings.PLAYER_DATA_PUBLIC_URL
            if crop_url.startswith(base_url):
                key = crop_url.split(settings.PLAYER_DATA_PUBLIC_URL)[-1].strip()[1:]
            else:
                key = crop_url
            pathfile = Path(INPUT_VIDEOS_DIR) / key
            pathfile.parent.mkdir(parents=True, exist_ok=True)
            r2_repo().download_player_image(key=key, destination_path=pathfile.as_posix())
            img = cv2.imread(pathfile.as_posix())
            pathfile.unlink(missing_ok=True)
            return img
        except Exception as e:
            logfire.error(f"[PlayerCropValidator] Error downloading image from {crop_url}: {e}")
            return None

    def _apply_merges_and_deletions(
        self,
        match_id: int,
        merge_map: Dict[int, int],
        to_delete: List[int],
        primary_ids: set,
        session: Session,
    ) -> None:
        stmt = select(PlayerModel).where(PlayerModel.match_id == match_id)
        all_players = session.exec(stmt).all()
        player_dict = {p.id: p for p in all_players}

        # Fusionar
        for secondary_id, primary_id in merge_map.items():
            if secondary_id not in player_dict or primary_id not in player_dict:
                logfire.warning(f"[PlayerCropValidator] Skipping merge: player {secondary_id} or {primary_id} not found")
                continue
            secondary = player_dict[secondary_id]
            primary = player_dict[primary_id]
            self._transfer_relations(secondary, primary, session)
            session.delete(secondary)
            logfire.info(f"[PlayerCropValidator] Merged player {secondary_id} into {primary_id}")

        # Eliminar inválidos (que no sean primarios)
        final_delete = [
            pid for pid in to_delete if pid not in primary_ids and pid in player_dict
        ]
        for pid in final_delete:
            player = player_dict.get(pid)
            if player:
                session.delete(player)
                logfire.info(f"[PlayerCropValidator] Deleted invalid player {pid}")

        session.commit()
        logfire.info(f"[PlayerCropValidator] Applied merges and deletions for match {match_id}")

    def _transfer_relations(self, source: PlayerModel, target: PlayerModel, session: Session):
        # Transferir número de camiseta si source tiene y target no
        if source.shirt_number is not None and target.shirt_number is None:
            target.shirt_number = source.shirt_number
            session.add(target)

        # Sumar estadísticas
        if source.goals > 0 or source.shots > 0 or source.team_goals > 0:
            target.goals += source.goals
            target.shots += source.shots
            target.team_goals += source.team_goals
            session.add(target)

        # Transferir relaciones (PlayerState, PlayerNumbers, DepthHistory)
        stmt = select(PlayerState).where(PlayerState.player_id == source.id)
        for s in session.exec(stmt).all():
            s.player_id = target.id
            session.add(s)

        stmt = select(PlayerNumbers).where(PlayerNumbers.player_id == source.id)
        for n in session.exec(stmt).all():
            n.player_id = target.id
            session.add(n)

        stmt = select(DepthHistory).where(DepthHistory.player_id == source.id)
        for d in session.exec(stmt).all():
            d.player_id = target.id
            session.add(d)

        session.flush()
        logfire.info(f"[PlayerCropValidator] Transferred relationships from player {source.id} to {target.id}")

    def _delete_players(self, player_ids: List[int], session: Session) -> None:
        for pid in player_ids:
            player = session.get(PlayerModel, pid)
            if player:
                session.delete(player)
        session.commit()
        logfire.info(f"[PlayerCropValidator] Deleted {len(player_ids)} players")

    # --------------------- Diagnóstico (calibración de umbrales) ---------------------

    def diagnose_threshold(self, player_data: list, known_duplicate_pairs: list):
        """
        Imprime distancias coseno, color y facial para pares conocidos, para calibrar umbrales.
        """
        id_to_idx = {d["player_id"]: i for i, d in enumerate(player_data)}

        print("=== Diagnóstico de distancias entre pares conocidos como duplicados ===")
        for pid_a, pid_b in known_duplicate_pairs:
            if pid_a not in id_to_idx or pid_b not in id_to_idx:
                print(f"  ({pid_a}, {pid_b}): uno de los dos no está en player_data")
                continue
            i, j = id_to_idx[pid_a], id_to_idx[pid_b]
            dist_cos = self._pair_distance(i, j, player_data)
            d1, d2 = player_data[i], player_data[j]
            color_dist = deltaE_ciede2000(d1["color_lab"], d2["color_lab"]) if (d1["color_lab"] is not None and d2["color_lab"] is not None) else None
            face_dist = cosine(d1["face_embedding"], d2["face_embedding"]) if (d1["face_embedding"] is not None and d2["face_embedding"] is not None) else None

            print(f"  player {pid_a} <-> {pid_b}: coseno={dist_cos:.4f}, color_CIEDE2000={color_dist if color_dist is not None else 'N/A'}, face_cos={face_dist if face_dist is not None else 'N/A'}")
        print("\nSugerencia: ajustar umbrales según los valores observados.")


player_crop_validator = PlayerCropValidator()