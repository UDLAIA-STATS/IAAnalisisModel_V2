from pathlib import Path

import cv2
import logfire
import numpy as np

from src.core.services.r2_manager import R2Manager
from src.entities.types.bucket_types import FilePurposeTypes


class R2Repository:
    def __init__(self):
        self.manager = R2Manager()

    
    def upload_player_image(self, key: str, crop: np.ndarray):
        success, encoded_image = cv2.imencode(".png", crop)

        if success:
            img_bytes = encoded_image.tobytes()
            url = self.manager.upload(key, img_bytes)
            return url
        
        logfire.error(f"[R2Repository] Error al subir la imagen del jugador: {key}")

        return ""

    def generate_key(self, match_id: int, filename: str, purpose_type: FilePurposeTypes, file_extension: str):
        return self.manager.generate_key(match_id, filename, purpose_type, file_extension)
    
    def get_public_url(self, key: str):
        return self.manager.get_public_url(key)

    def upload_heatmap(self, key: str, heatmap: Path):
        return self.manager.upload(key, heatmap.read_bytes())


    def upload_report(self, key: str, report: Path):
        return self.manager.upload(key, report.read_bytes(), "application/json")


    def steam_download(self, key: str, destination_path: str, chunk_size=1024 * 1024 * 16):
        return self.manager.stream_download(key, destination_path, chunk_size)


files_repository = R2Repository()
