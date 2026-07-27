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
from src.config.routes import INPUT_VIDEOS_DIR, PLAYER_MODEL_PATH
from src.config.configuration import settings

# Color recognizer
from src.core.vision.color_recognizer import jersey_color_extractor
from skimage.color import deltaE_ciede2000, rgb2lab


class PlayerCropValidator:
    """
    Valida y unifica jugadores a partir de sus imágenes recortadas (crop_path).

    Pipeline de validación (por jugador):

      1) PRESENCIA / COMPLETITUD -> puede ELIMINAR el crop
         - Etapa primaria: detección de persona con YOLO.
         - Etapa de respaldo (solo se ejecuta si YOLO no detecta nada):
           segmentación por regiones (MSER). Si se encuentran más de
           MIN_SEGMENTS_FOR_COMPLETENESS regiones, se considera que el
           jugador está presente y visible por completo.
         - El crop SOLO se elimina si AMBAS etapas fallan (ni YOLO ni la
           segmentación de respaldo detectan al jugador). Si cualquiera de
           las dos tiene éxito, el crop se conserva.

      2) SIMILITUD / FUSIÓN -> nunca elimina, solo combina o deja igual
         - Dos jugadores se fusionan (se combinan en un único registro)
           únicamente si:
             a) Sus embeddings son similares (distancia coseno).
             b) Sus colores de uniforme son similares (deltaE CIEDE2000).
             c) Sus números de camiseta son compatibles (iguales, o alguno
                de los dos es None).
         - Si el número de camiseta difiere entre A y B -> NO se fusionan
           y TAMPOCO se eliminan: quedan como jugadores independientes.
         - Si no cumplen los 3 criterios de similitud -> tampoco se
           fusionan ni se eliminan.
    """

    YOLO_CONFIDENCE_THRESHOLD: float = 0.15
    COSINE_DISTANCE_THRESHOLD: float = 0.15
    MIN_PLAYERS_TO_CLUSTER: int = 3
    MIN_SEGMENTS_FOR_COMPLETENESS: int = 3  # "más de 3 segmentos" -> completo
    COLOR_SIMILARITY_THRESHOLD: float = 10.0

    YOLO_MODEL_PATH: str = PLAYER_MODEL_PATH.as_posix()

    def __init__(self, reid_model_name: str = "osnet_x1_0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logfire.info(f"[PlayerCropValidator] Using device: {self.device}")

        # BUGFIX: antes se ignoraba YOLO_MODEL_PATH (definido desde config)
        # y siempre se cargaba "yolo26x.pt" hardcodeado. Ahora se respeta
        # la configuración del modelo de jugadores.
        self.yolo = YOLO(self.YOLO_MODEL_PATH)
        self.reid_model = self._load_reid_model(reid_model_name)
        self.color_extractor = jersey_color_extractor

    # --------------------- Utilidades de imagen ---------------------

    def _estimate_blur(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

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
            logfire.warning(
                "[PlayerCropValidator] Using ResNet50 as fallback ReID model"
            )
            return model

    # --------------------- Condición 1: presencia/completitud ---------------------

    def _detect_person_yolo(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float, float, float]], float]:
        """Etapa primaria: detección de persona con YOLO."""
        results = self.yolo.predict(
            image,
            conf=0.15,
            verbose=False,
            device=self.device,
            agnostic_nms=True,
            end2end=True,
        )

        max_conf = 0.0
        person_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    person_boxes.append((x1, y1, x2, y2, conf))

        return person_boxes, max_conf

    def _detect_segmentation_backup(
        self, image: np.ndarray
    ) -> Tuple[List, float, bool]:
        """
        Etapa de respaldo (solo si YOLO no detecta nada). Cuenta regiones
        mediante MSER; si hay más de MIN_SEGMENTS_FOR_COMPLETENESS regiones
        válidas, se considera que el jugador está presente y visible por
        completo en la imagen.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER.create()
        regions, _ = mser.detectRegions(gray)
        min_area = 100
        valid_regions = [r for r in regions if r.shape[0] >= min_area]

        if len(valid_regions) > self.MIN_SEGMENTS_FOR_COMPLETENESS:
            h, w, _ = image.shape
            box = (0, 0, w, h, 0.5)
            logfire.info(
                f"[PlayerCropValidator] Segmentación de respaldo: {len(valid_regions)} "
                f"regiones (> {self.MIN_SEGMENTS_FOR_COMPLETENESS}), jugador considerado completo."
            )
            return [box], 0.5, True

        logfire.info(
            f"[PlayerCropValidator] Segmentación de respaldo: solo {len(valid_regions)} "
            f"regiones, jugador incompleto/no detectado."
        )
        return [], 0.0, False

    def _check_presence_and_completeness(
        self, image: np.ndarray
    ) -> Tuple[bool, float, str]:
        """
        Condición 1 completa: presencia y completitud del jugador.

        - Si YOLO detecta con conf >= YOLO_CONFIDENCE_THRESHOLD -> válido.
        - Si YOLO falla, se intenta la segmentación de respaldo (MSER).
          Si supera MIN_SEGMENTS_FOR_COMPLETENESS regiones -> válido.
        - Se elimina el crop SOLO si ambas etapas fallan.

        Retorna (es_valido, confianza, metodo_usado).
        """
        blur_score = self._estimate_blur(image)
        enhanced_image = self._sharpen_image(image) if blur_score < 50.0 else image

        _, yolo_conf = self._detect_person_yolo(enhanced_image)
        if yolo_conf >= self.YOLO_CONFIDENCE_THRESHOLD:
            return True, yolo_conf, "yolo"

        _, seg_conf, seg_ok = self._detect_segmentation_backup(enhanced_image)
        if seg_ok:
            return True, seg_conf, "segmentation_backup"

        return False, 0.0, "none"

    # --------------------- Extracción de features ---------------------

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

    def _colors_are_similar(
        self,
        lab1: np.ndarray,
        lab2: np.ndarray,
        threshold: float = COLOR_SIMILARITY_THRESHOLD,
    ) -> bool:
        if lab1 is None or lab2 is None:
            return False
        try:
            delta_e = deltaE_ciede2000(lab1, lab2)
            return delta_e < threshold
        except Exception as e:
            logfire.warning(
                f"[PlayerCropValidator] Error calculating color distance: {e}"
            )
            return False

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
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
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    # --------------------- Condición 2: similitud/fusión ---------------------

    def _cluster_embeddings_and_color(self, player_data: List[Dict]) -> List[List[int]]:
        """
        Condición 2: agrupa primero por embeddings (distancia coseno), luego
        subdivide cada grupo por color de uniforme y número de camiseta.

        Dos jugadores se fusionan solo si, simultáneamente:
          - Son similares en embedding (vectorialmente).
          - Tienen colores de uniforme similares (ambos detectados y
            deltaE < COLOR_SIMILARITY_THRESHOLD).
          - Sus números de camiseta son compatibles: ambos None, uno None,
            o iguales. Si ambos números están presentes y difieren, NUNCA
            se fusionan (quedan en clusters/grupos distintos).

        Esta función solo decide fusiones; nunca marca jugadores para
        eliminación (eso corresponde exclusivamente a la Condición 1).
        """
        if len(player_data) < 2:
            return [[d["player_id"] for d in player_data]]

        ids = [d["player_id"] for d in player_data]
        embeddings = np.array([d["embedding"] for d in player_data])
        colors_lab = [d.get("color_lab") for d in player_data]
        shirt_numbers = [d.get("shirt_number") for d in player_data]

        dist_matrix = pdist(embeddings, metric="cosine")
        Z = linkage(dist_matrix, method="average")
        clusters = fcluster(Z, self.COSINE_DISTANCE_THRESHOLD, criterion="distance")
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            cluster_dict.setdefault(cluster_id, []).append(idx)

        final_clusters = []

        for indices in cluster_dict.values():
            if len(indices) == 1:
                final_clusters.append([ids[i] for i in indices])
                continue

            remaining = set(indices)
            while remaining:
                ref_idx = next(iter(remaining))
                ref_lab = colors_lab[ref_idx]
                ref_num = shirt_numbers[ref_idx]
                group = [ref_idx]
                remaining.remove(ref_idx)

                to_remove = []
                for idx in remaining:
                    other_lab = colors_lab[idx]
                    other_num = shirt_numbers[idx]

                    color_ok = (
                        ref_lab is not None
                        and other_lab is not None
                        and self._colors_are_similar(ref_lab, other_lab)
                    )
                    # Números de camiseta incompatibles (ambos presentes y
                    # distintos) bloquean la fusión sin importar lo demás.
                    num_ok = (
                        ref_num is None or other_num is None or ref_num == other_num
                    )

                    if color_ok and num_ok:
                        group.append(idx)
                        to_remove.append(idx)

                for idx in to_remove:
                    remaining.remove(idx)

                final_clusters.append([ids[i] for i in group])

        return final_clusters

    def _select_primary_player(
        self, cluster_ids: List[int], player_data: List[Dict]
    ) -> int:
        return max(
            cluster_ids,
            key=lambda pid: next(
                d["conf"] for d in player_data if d["player_id"] == pid
            ),
        )

    # --------------------- Orquestación principal ---------------------

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

                # Condición 1: presencia y completitud (YOLO -> respaldo MSER).
                # Solo se elimina si ambas etapas fallan.
                is_present, conf, method = self._check_presence_and_completeness(image)
                if not is_present:
                    logfire.info(
                        f"[PlayerCropValidator] Player {player.id}: no se detectó jugador "
                        f"(YOLO y segmentación de respaldo fallaron), marcando para eliminación"
                    )
                    to_delete_ids.append(player.id)
                    continue

                embedding = self._extract_embedding(image)
                color_rgb, color_hex = self._extract_color(image)
                color_lab = self._rgb_to_lab(color_rgb)

                player_data.append(
                    {
                        "player_id": player.id,
                        "embedding": embedding,
                        "track_id": player.track_id,
                        "conf": conf,
                        "color_rgb": color_rgb,
                        "color_lab": color_lab,
                        "color_hex": color_hex,
                        "shirt_number": player.shirt_number,
                    }
                )
                logfire.info(
                    f"[PlayerCropValidator] Player {player.id}: jugador válido "
                    f"(método={method}, conf={conf:.2f}), embedding y color extraídos"
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

        # Condición 2: fusión por similitud (no genera eliminaciones nuevas).
        clusters = self._cluster_embeddings_and_color(player_data)

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

    # --------------------- Persistencia ---------------------

    def _download_image(self, crop_url: str) -> Optional[np.ndarray]:
        try:
            base_url = settings.PLAYER_DATA_PUBLIC_URL
            if crop_url.startswith(base_url):
                key = crop_url.split(settings.PLAYER_DATA_PUBLIC_URL)[-1].strip()[1:]
            else:
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

        # Solo se eliminan los IDs que fallaron la Condición 1 (presencia/
        # completitud) y que no terminaron siendo el jugador primario de
        # un cluster.
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
        if (
            source.shirt_number != target.shirt_number
            and source.shirt_number is not None
        ):
            target.shirt_number = source.shirt_number
            session.add(target)
            logfire.info(f"[PlayerCropValidator] Updated player {target.id}")
        if source.goals > 0 or source.shots > 0 or source.team_goals > 0:
            target.goals += source.goals
            target.shots += source.shots
            target.team_goals += source.team_goals
            session.add(target)
            logfire.info(f"[PlayerCropValidator] Updated player {target.id}")
        stmt = select(PlayerState).where(PlayerState.player_id == source.id)
        states = session.exec(stmt).all()
        for s in states:
            s.player_id = target.id
            session.add(s)
        stmt = select(PlayerNumbers).where(PlayerNumbers.player_id == source.id)
        nums = session.exec(stmt).all()
        for n in nums:
            n.player_id = target.id
            session.add(n)
        stmt = select(DepthHistory).where(DepthHistory.player_id == source.id)
        depths = session.exec(stmt).all()
        for d in depths:
            d.player_id = target.id
            session.add(d)
        logfire.info(
            f"[PlayerCropValidator] Transferred relationships from player {source.id} to {target.id}"
        )
        session.flush()

    def _delete_players(self, player_ids: List[int], session: Session) -> None:
        for pid in player_ids:
            player = session.get(PlayerModel, pid)
            if player:
                session.delete(player)
        session.commit()
        logfire.info(f"[PlayerCropValidator] Deleted {len(player_ids)} players")


player_crop_validator = PlayerCropValidator()
