import uuid

import boto3
from botocore.client import Config as BotoConfig

from shared.config import get_settings

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        settings = get_settings()

        # Cloudflare R2 Object Storage credentials
        r2_access_key = settings.get("R2_ACCESS_KEY")
        r2_secret_key = settings.get("R2_SECRET_KEY")
        r2_account_id = settings.get("R2_ACCOUNT_ID")
        r2_endpoint = settings.get("R2_ENDPOINT_URL")

        if r2_access_key and r2_secret_key:
            endpoint_url = r2_endpoint
            if not endpoint_url and r2_account_id:
                endpoint_url = f"https://{r2_account_id.strip()}.r2.cloudflarestorage.com"

            _s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=r2_access_key,
                aws_secret_access_key=r2_secret_key,
                region_name=settings.get("R2_REGION", "auto") or "auto",
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
            )
        else:
            # Fallback to AWS S3 Storage
            region = settings.get("AWS_REGION") or "eu-north-1"
            _s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.get("AWS_ACCESS_KEY"),
                aws_secret_access_key=settings.get("AWS_ACCESS_SECRET"),
                region_name=region,
                config=BotoConfig(signature_version="s3v4"),
            )
    return _s3_client


def get_bucket_name() -> str:
    settings = get_settings()
    return settings.get("R2_BUCKET") or settings.get("AWS_BUCKET", "agentlytics")


def build_upload_key(workspace_id: str, file_id: str, filename: str) -> str:
    return f"workspaces/{workspace_id}/uploads/{file_id}/{filename}"


def build_chart_key(workspace_id: str, chart_id: str) -> str:
    return f"workspaces/{workspace_id}/charts/{chart_id}.html"


def build_report_key(workspace_id: str, report_id: str, ext: str = "html") -> str:
    return f"workspaces/{workspace_id}/reports/{report_id}.{ext.lstrip('.')}"


def presign_put(key: str, expires_in: int = 3600) -> str:
    return get_s3_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": get_bucket_name(), "Key": key},
        ExpiresIn=expires_in,
    )


def presign_get(
    key: str,
    expires_in: int = 3600,
    download_name: str | None = None,
    inline: bool = False,
    content_type: str | None = None,
) -> str:
    params = {"Bucket": get_bucket_name(), "Key": key}
    if download_name:
        safe_name = download_name.replace('"', "").replace("\\", "")
        disposition = "inline" if inline else "attachment"
        params["ResponseContentDisposition"] = f'{disposition}; filename="{safe_name}"'
    if content_type:
        params["ResponseContentType"] = content_type
    return get_s3_client().generate_presigned_url("get_object", Params=params, ExpiresIn=expires_in)


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
