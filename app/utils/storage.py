"""Artifact storage (local FS or S3) with uniform signed-URL surface."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import quote

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("utils.storage")


class StorageClient:
    """
    Persist a final artifact and return a URL the caller can fetch.

    backend=local → copied to STORAGE_LOCAL_DIR, `file://` URL returned.
    backend=s3    → uploaded to S3_BUCKET/S3_PREFIX, pre-signed URL returned.
    """

    def upload(self, src: Path, key: str, *, content_type: str | None = None) -> str:
        if settings.storage.backend == "s3":
            return self._upload_s3(src, key, content_type)
        return self._upload_local(src, key)

    def _upload_local(self, src: Path, key: str) -> str:
        dest = settings.storage.local_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        logger.info("stored artifact locally: %s", dest)
        return dest.resolve().as_uri()

    def _upload_s3(self, src: Path, key: str, content_type: str | None) -> str:
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("boto3 not installed; add it to requirements.txt for S3 storage") from exc

        bucket = settings.storage.s3_bucket
        if not bucket:
            raise RuntimeError("S3_BUCKET is empty while STORAGE_BACKEND=s3")

        full_key = f"{settings.storage.s3_prefix.rstrip('/')}/{quote(key)}"
        extra: dict = {}
        if content_type:
            extra["ContentType"] = content_type

        s3 = boto3.client("s3", region_name=settings.storage.s3_region)
        s3.upload_file(str(src), bucket, full_key, ExtraArgs=extra or None)
        logger.info("uploaded s3://%s/%s", bucket, full_key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": full_key},
            ExpiresIn=settings.storage.s3_signed_url_ttl,
        )
        return url


storage_client = StorageClient()
