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
from scipy.spatial.distance import cosine

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
    COSINE_DISTANCE_THRESHOLD: float = 0.15  # ya no se usa en el clustering, la dejo por si la referenciás en otro lado
    MIN_PLAYERS_TO_CLUSTER: int = 3
    MIN_SEGMENTS_FOR_COMPLETENESS: int = 3  # "más de 3 segmentos" -> completo
    COLOR_SIMILARITY_THRESHOLD: float = 10.0
    EMBEDDING_WEIGHT: float = 0.6
    COLOR_WEIGHT: float = 0.4
    COMBINED_DISTANCE_THRESHOLD: float = 0.35
    INCOMPATIBLE_NUMBER_DISTANCE: float = 999.0

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

    
def _combined_pair_distance(
    self,
    i: int,
    j: int,
    embeddings: list,
    colors_lab: list,
    shirt_numbers: list,
) -> float:
    """
    Distancia combinada entre dos jugadores (índices i, j dentro de
    player_data), mezclando embedding + color, con bloqueo duro por
    número de camiseta incompatible.
 
    - Si ambos números de camiseta están presentes y son distintos ->
      distancia gigante (nunca se fusionan), igual que en la lógica
      original.
    - Si alguno de los dos números es None (o ambos) -> el número no
      bloquea nada, se decide solo por embedding + color.
    - El color solo pesa si AMBOS colores fueron extraídos con éxito;
      si falta alguno, se usa una penalización neutra (ni ayuda ni
      perjudica demasiado la fusión) para no castigar crops donde el
      extractor de color falló pero el jugador es claramente el mismo
      por embedding.
    """
    num_i = shirt_numbers[i]
    num_j = shirt_numbers[j]
    if num_i is not None and num_j is not None and num_i != num_j:
        return self.INCOMPATIBLE_NUMBER_DISTANCE
 
    emb_dist = cosine(embeddings[i], embeddings[j])
 
    lab_i = colors_lab[i]
    lab_j = colors_lab[j]
    if lab_i is not None and lab_j is not None:
        try:
            # deltaE CIEDE2000 típicamente está en rango 0-100+, se
            # normaliza a una escala comparable con la distancia coseno
            # (0-2 aprox, casi siempre 0-1 en la práctica).
            color_dist = float(deltaE_ciede2000(lab_i, lab_j)) / 100.0
        except Exception:
            color_dist = 0.5
    else:
        color_dist = 0.5  # penalización neutra, color desconocido
 
    return self.EMBEDDING_WEIGHT * emb_dist + self.COLOR_WEIGHT * color_dist
 
 
def _build_combined_distance_matrix(self, player_data: list) -> np.ndarray:
    """
    Construye la matriz de distancia condensada (formato que espera
    scipy.cluster.hierarchy.linkage) combinando embedding + color +
    número de camiseta para todos los pares de jugadores.
    """
    n = len(player_data)
    embeddings = [d["embedding"] for d in player_data]
    colors_lab = [d.get("color_lab") for d in player_data]
    shirt_numbers = [d.get("shirt_number") for d in player_data]
 
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(
                self._combined_pair_distance(
                    i, j, embeddings, colors_lab, shirt_numbers
                )
            )
    return np.array(condensed)
 
 
def _cluster_embeddings_and_color(self, player_data: list) -> list:
    """
    Condición 2 (versión combinada): agrupa jugadores en un único paso
    de clustering jerárquico sobre una distancia que mezcla embedding,
    color y número de camiseta. Reemplaza el enfoque anterior de
    "clusterizar por embedding y subdividir después", que perdía
    fusiones válidas cuando el embedding por sí solo no alcanzaba el
    umbral aunque color y número coincidieran.
 
    Nunca elimina jugadores; solo decide qué IDs van juntos en cada
    cluster (eso lo sigue haciendo exclusivamente la Condición 1).
    """
    if len(player_data) < 2:
        return [[d["player_id"] for d in player_data]]
 
    ids = [d["player_id"] for d in player_data]
    condensed = self._build_combined_distance_matrix(player_data)
 
    Z = linkage(condensed, method="average")
    cluster_labels = fcluster(
        Z, self.COMBINED_DISTANCE_THRESHOLD, criterion="distance"
    )
 
    cluster_dict = {}
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_dict.setdefault(cluster_id, []).append(ids[idx])
 
    return list(cluster_dict.values())
 
 
# --------------------- Herramienta de calibración (opcional) ---------------------
 
def diagnose_threshold(self, player_data: list, known_duplicate_pairs: list):
    """
    Función de diagnóstico, NO para producción. Sirve para calibrar
    COMBINED_DISTANCE_THRESHOLD con casos reales.
 
    Uso:
        # known_duplicate_pairs: lista de tuplas (player_id_a, player_id_b)
        # que vos SABÉS que son el mismo jugador duplicado en la BD.
        validator.diagnose_threshold(player_data, [(101, 205), (102, 340)])
 
    Imprime la distancia combinada real para cada par conocido, para que
    puedas fijar el umbral apenas por encima del valor más alto observado
    entre duplicados verdaderos (y confirmar que sigue por debajo de la
    distancia entre jugadores distintos).
    """
    id_to_idx = {d["player_id"]: i for i, d in enumerate(player_data)}
    embeddings = [d["embedding"] for d in player_data]
    colors_lab = [d.get("color_lab") for d in player_data]
    shirt_numbers = [d.get("shirt_number") for d in player_data]
 
    print("=== Distancias combinadas entre pares CONOCIDOS como duplicados ===")
    for pid_a, pid_b in known_duplicate_pairs:
        if pid_a not in id_to_idx or pid_b not in id_to_idx:
            print(f"  ({pid_a}, {pid_b}): uno de los dos no está en player_data, se saltea")
            continue
        i, j = id_to_idx[pid_a], id_to_idx[pid_b]
        dist = self._combined_pair_distance(i, j, embeddings, colors_lab, shirt_numbers)
        emb_dist = cosine(embeddings[i], embeddings[j])
        print(
            f"  player {pid_a} <-> player {pid_b}: "
            f"combinada={dist:.4f}  (embedding_solo={emb_dist:.4f})"
        )
    print(
        "\nSugerencia: fijá COMBINED_DISTANCE_THRESHOLD un poco por encima "
        "del valor más alto de 'combinada' que veas arriba entre duplicados reales."
    )

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
