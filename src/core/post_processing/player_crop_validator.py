import tempfile
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
from src.entities.models.soccer.player_model import PlayerModel, PlayerState, PlayerNumbers, DepthHistory
from src.config.routes import PLAYER_MODEL_PATH
from src.config.configuration import settings


class PlayerCropValidator:
    """
    Valida y unifica jugadores a partir de sus imágenes recortadas (crop_path).
    """

    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    COSINE_DISTANCE_THRESHOLD: float = 0.4
    MIN_PLAYERS_TO_CLUSTER: int = 2

    YOLO_MODEL_PATH: str = PLAYER_MODEL_PATH.as_posix()

    def __init__(self, reid_model_name: str = "osnet_x1_0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logfire.info(f"[PlayerCropValidator] Using device: {self.device}")

        self.yolo = YOLO(self.YOLO_MODEL_PATH)
        self.reid_model = self._load_reid_model(reid_model_name)

    def _load_reid_model(self, model_name: str):
        """Carga modelo pre-entrenado de torchreid."""
        try:
            from torchreid import models
            model = models.build_model(
                name=model_name,
                num_classes=1000,
                pretrained=True
            )
            model = model.model

            if hasattr(model, 'classifier'):
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

                _, max_conf = self._detect_person(image)
                if max_conf < self.YOLO_CONFIDENCE_THRESHOLD:
                    logfire.info(f"[PlayerCropValidator] Player {player.id}: no person detected (conf={max_conf:.2f}), marking for deletion")
                    to_delete_ids.append(player.id)
                    continue

                embedding = self._extract_embedding(image)
                player_data.append({
                    "player_id": player.id,
                    "embedding": embedding,
                    "track_id": player.track_id,
                    "conf": max_conf,
                })
                logfire.info(f"[PlayerCropValidator] Player {player.id}: valid person detected, embedding extracted")

            except Exception as e:
                logfire.error(f"[PlayerCropValidator] Error processing player {player.id}: {e}")
                to_delete_ids.append(player.id)

        if len(player_data) < self.MIN_PLAYERS_TO_CLUSTER:
            logfire.info("[PlayerCropValidator] Not enough valid players to cluster, only deleting invalid ones")
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
            match_id,
            merge_map,
            to_delete_ids,
            primary_players,
            session
        )

        logfire.info(f"[PlayerCropValidator] Validation completed for match {match_id}")

    def _download_image(self, crop_url: str) -> Optional[np.ndarray]:
        """
        Descarga la imagen desde R2 usando el repositorio R2Repository.
        """
        try:
            base_url = settings.PLAYER_DATA_PUBLIC_URL
            if crop_url.startswith(base_url):
                key = crop_url[len(base_url):].lstrip('/')
            else:
                # Si no coincide, intentar usar la URL directamente (podría ser key)
                key = crop_url

            # Descargar a un archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                r2_repo().steam_download(key=key, destination_path=str(tmp_path))
                img = cv2.imread(str(tmp_path))
                tmp_path.unlink(missing_ok=True)
                return img

        except Exception as e:
            logfire.error(f"[PlayerCropValidator] Error downloading image from {crop_url}: {e}")
            return None

    def _detect_person(self, image: np.ndarray) -> Tuple[List, float]:
        """
        Usa YOLO para detectar personas en la imagen.
        Retorna: (lista de bboxes, confianza máxima)
        """
        results = self.yolo(image, verbose=False)
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
        return person_boxes, max_conf

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen y extrae embedding usando el modelo ReID.
        """
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
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
        dist_matrix = pdist(embeddings, metric='cosine')
        # Linkage jerárquico (average linkage)
        Z = linkage(dist_matrix, method='average')
        # Cortar por umbral de distancia
        clusters = fcluster(Z, self.COSINE_DISTANCE_THRESHOLD, criterion='distance')
        # Agrupar IDs por cluster
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            cluster_dict.setdefault(cluster_id, []).append(ids[idx])
        return list(cluster_dict.values())

    def _select_primary_player(self, cluster_ids: List[int], player_data: List[Dict]) -> int:
        """
        Elige el jugador principal del cluster (el que tiene mayor confianza de detección).
        """
        best = max(cluster_ids, key=lambda pid: next(d["conf"] for d in player_data if d["player_id"] == pid))
        return best

    def _apply_merges_and_deletions(
        self,
        match_id: int,
        merge_map: Dict[int, int],
        to_delete: List[int],
        primary_ids: set,
        session: Session
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
                logfire.warning(f"[PlayerCropValidator] Skipping merge: player {secondary_id} or {primary_id} not found")
                continue
            secondary = player_dict[secondary_id]
            primary = player_dict[primary_id]

            self._transfer_relations(secondary, primary, session)
            session.delete(secondary)
            logfire.info(f"[PlayerCropValidator] Merged player {secondary_id} into {primary_id}")

        # 2. Eliminar jugadores inválidos (los que no tienen persona y no son primary)
        final_delete = [pid for pid in to_delete if pid not in primary_ids and pid in player_dict]
        for pid in final_delete:
            player = player_dict.get(pid)
            if player:
                session.delete(player)
                logfire.info(f"[PlayerCropValidator] Deleted invalid player {pid}")

        session.commit()
        logfire.info(f"[PlayerCropValidator] Applied merges and deletions for match {match_id}")

    def _transfer_relations(self, source: PlayerModel, target: PlayerModel, session: Session):
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
        logfire.info(f"[PlayerCropValidator] Transferred relationships from player {source.id} to {target.id}")

    def _delete_players(self, player_ids: List[int], session: Session) -> None:
        """Elimina jugadores de la base de datos (cascada automática si está configurada)."""
        for pid in player_ids:
            player = session.get(PlayerModel, pid)
            if player:
                session.delete(player)
        session.commit()
        logfire.info(f"[PlayerCropValidator] Deleted {len(player_ids)} players")

player_crop_validator = PlayerCropValidator()