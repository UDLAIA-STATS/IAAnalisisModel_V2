from io import BytesIO
from pathlib import Path
import traceback
import logfire

from tenacity import retry, stop_after_attempt, wait_exponential, wait_incrementing
from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

from src.entities.services.r2_manager_base import R2ManagerBase
from src.entities.types.bucket_types import FilePurposeTypes


class R2Manager(R2ManagerBase):
    def __init__(self):
        super().__init__()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.2, min=4, max=8))
    def upload(
        self,
        match_id: int,
        file_bytes: bytes,
        filename: str,
        file_extension: str,
        purpose_type: FilePurposeTypes,
        file_type: str = "image/png"):
        try:
            bucket = self.artefactos_bucket
            key = self.generate_key(
                match_id=match_id,
                filename=filename,
                purpose_type=purpose_type,
                file_extension=file_extension,
            )

            self.client.upload_fileobj(
                Fileobj=BytesIO(file_bytes),
                Bucket=bucket,
                Key=key,
                ExtraArgs={"ContentType": file_type},
            )

            logfire.info(f"Archivo subido correctamente: {filename}")

        except Exception as e:
            logfire.error(f"Error subiendo archivo: {traceback.format_exc()}")
            raise e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_incrementing(start=0, increment=2, max=10),
        reraise=True,
        )
    def stream_download(
        self, key: str, destination_path: str, chunk_size=1024 * 1024 * 16
    ):
        """
        Descarga el archivo en chunks (16 MB por defecto).
        Soporta archivos grandes (+5GB).
        """
        try:
            logfire.info(f"[R2 Manager] Descargando {key} a {destination_path}...")
            with open(destination_path, "wb") as f:
                obj = self.client.get_object(Bucket=self.video_bucket, Key=key)
                body = obj["Body"]

                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        logfire.info("[R2 Manager] Descarga completada.")
                        break

                    f.write(chunk)
                    f.flush()
        except (BotoCoreError, ClientError, ReadTimeoutError) as e:
            logfire.error(f"Error descargando {key} desde R2: {e}")
            Path(destination_path).unlink(missing_ok=True)
            raise e
        except Exception as e:
            logfire.error(f"Error descargando {key} desde R2: {e}")
            Path(destination_path).unlink(missing_ok=True)
            raise e
