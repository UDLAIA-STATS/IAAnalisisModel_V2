import boto3
from botocore.config import Config

from src.config.configuration import settings


class R2Manager:
    def __init__(self):
        config = Config(
            read_timeout=800,
            connect_timeout=50,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.VIDEOS_S3_ENDPOINT,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            config=config,
        )
        self.bucket = settings.VIDEO_BUCKET
