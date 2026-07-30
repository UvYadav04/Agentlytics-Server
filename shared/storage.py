import uuid

import boto3
from botocore.client import Config as BotoConfig

from shared.config import get_settings

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        settings = get_settings()
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.get("AWS_ACCESS_KEY"),
            aws_secret_access_key=settings.get("AWS_ACCESS_SECRET"),
            region_name=settings.get("AWS_REGION"),
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3_client


def get_bucket_name() -> str:
    return get_settings().get("AWS_BUCKET", "data-analyzer")


def build_upload_key(workspace_id: str, file_id: str, filename: str) -> str:
    return f"workspaces/{workspace_id}/uploads/{file_id}/{filename}"


def build_chart_key(workspace_id: str, chart_id: str) -> str:
    return f"workspaces/{workspace_id}/charts/{chart_id}.html"


def build_report_key(workspace_id: str, report_id: str, ext: str = "html") -> str:
    return f"workspaces/{workspace_id}/reports/{report_id}.{ext.lstrip('.')}"


def presign_put(key: str, content_type: str = "application/octet-stream", expires_in: int = 3600) -> str:
    return get_s3_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": get_bucket_name(), "Key": key, "ContentType": content_type},
        ExpiresIn=expires_in,
    )


def presign_get(key: str, expires_in: int = 3600) -> str:
    return get_s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": get_bucket_name(), "Key": key},
        ExpiresIn=expires_in,
    )


def delete_object(key: str) -> None:
    get_s3_client().delete_object(Bucket=get_bucket_name(), Key=key)


def object_exists(key: str) -> bool:
    try:
        get_s3_client().head_object(Bucket=get_bucket_name(), Key=key)
        return True
    except Exception:
        return False


def new_file_id() -> str:
    return uuid.uuid4().hex
