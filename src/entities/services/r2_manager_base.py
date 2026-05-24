from abc import ABC, abstractmethod
import uuid

import boto3
from botocore.config import Config

from src.entities.types.bucket_types import FilePurposeTypes
from src.config.configuration import settings


class R2ManagerBase(ABC):
    def __init__(self):
        config = Config(
            read_timeout=800,
            connect_timeout=60,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self.s3_endpoint = settings.VIDEOS_S3_ENDPOINT
        self.client = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            config=config,
        )
        self.video_bucket = settings.VIDEO_BUCKET
        self.artefactos_bucket = settings.ARTEFACTOS_BUCKET

        self.artefactos_public_url = settings.ARTEFACTOS_PUBLIC_URL

    def generate_key(
            self,
            match_id: int,
            filename: str,
            purpose_type: FilePurposeTypes,
            file_extension: str
    ) -> str:
        bucket = self.artefactos_bucket
        if purpose_type == FilePurposeTypes.PLAYER_IMAGE:
            return f"{bucket}/{match_id}/players/{filename}_{uuid.uuid4()}.{file_extension}"
        elif purpose_type == FilePurposeTypes.HEATMAP:
            return f"{bucket}/{match_id}/heatmaps/{filename}_{uuid.uuid4()}.{file_extension}"
        elif purpose_type == FilePurposeTypes.REPORTS:
            return f"{bucket}/{match_id}/reports/{filename}_{uuid.uuid4()}.{file_extension}"

        return f"{bucket}/{match_id}/{filename}_{uuid.uuid4()}.{file_extension}"


    @abstractmethod
    def upload(
            self,
            match_id: int,
            file_bytes: bytes,
            filename: str,
            file_extension: str,
            purpose_type: FilePurposeTypes,
            file_type: str = "image/png",
    ):
        pass

    @abstractmethod
    def stream_download(
        self, key: str, destination_path: str, chunk_size=1024 * 1024 * 16
    ):
        pass