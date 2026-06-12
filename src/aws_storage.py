"""AWS S3 helpers for model and experiment artifact storage.

Credentials are read from environment variables. They are never hard-coded or
included in console output.
"""

import os
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # Keep local experiments usable without optional cloud setup.
    boto3 = None

    class BotoCoreError(Exception):
        """Fallback exception type used when botocore is unavailable."""

    class ClientError(Exception):
        """Fallback exception type used when botocore is unavailable."""


MODEL_SUFFIXES = {".joblib", ".pkl", ".h5", ".keras"}


def _warn(message: str) -> None:
    """Print a friendly AWS warning without exposing credentials."""
    print(f"[AWS WARNING] {message}")


def _normalize_s3_key(s3_key: str) -> str:
    """Return a portable S3 object key without a leading slash."""
    normalized = str(PurePosixPath(str(s3_key).replace("\\", "/"))).lstrip("/")
    if normalized in {"", "."}:
        raise ValueError("s3_key cannot be empty.")
    return normalized


def is_s3_configured(
    bucket_name: str | None = None,
    *,
    warn: bool = True,
) -> bool:
    """Check required environment variables without printing their values."""
    missing: list[str] = []
    if not bucket_name:
        missing.append("AWS_S3_BUCKET_NAME")
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        missing.append("AWS_ACCESS_KEY_ID")
    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        missing.append("AWS_SECRET_ACCESS_KEY")
    if boto3 is None:
        missing.append("boto3")

    if missing and warn:
        _warn(
            "S3 operation skipped because configuration is missing: "
            f"{', '.join(missing)}."
        )
    return not missing


def _create_s3_client():
    """Create an S3 client using environment variables only."""
    if boto3 is None:
        return None

    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_DEFAULT_REGION") or None,
    )


def _print_aws_error(action: str, error: Exception) -> None:
    """Print a sanitized AWS error that never includes credential values."""
    if isinstance(error, ClientError):
        error_data = error.response.get("Error", {})
        code = error_data.get("Code", "ClientError")
        _warn(
            f"{action} failed [{code}]. "
            "Check the bucket, region, credentials, and IAM permissions."
        )
        return

    _warn(f"{action} failed ({type(error).__name__}).")


def upload_file_to_s3(
    local_path: str | Path,
    bucket_name: str | None,
    s3_key: str,
) -> bool:
    """Upload one local file to S3 and return whether it succeeded."""
    source = Path(local_path)
    if not source.exists() or not source.is_file():
        _warn(f"Upload skipped because local file does not exist: {source}")
        return False
    if not is_s3_configured(bucket_name):
        return False

    try:
        normalized_key = _normalize_s3_key(s3_key)
        client = _create_s3_client()
        if client is None:
            return False
        client.upload_file(str(source), str(bucket_name), normalized_key)
        return True
    except (BotoCoreError, ClientError, OSError, ValueError) as exc:
        _print_aws_error(f"Upload of {source.name}", exc)
        return False


def download_file_from_s3(
    bucket_name: str | None,
    s3_key: str,
    local_path: str | Path,
) -> bool:
    """Download one S3 object to a local file and return success status."""
    if not is_s3_configured(bucket_name):
        return False

    destination = Path(local_path)
    try:
        normalized_key = _normalize_s3_key(s3_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        client = _create_s3_client()
        if client is None:
            return False
        client.download_file(str(bucket_name), normalized_key, str(destination))
        return True
    except (BotoCoreError, ClientError, OSError, ValueError) as exc:
        _print_aws_error(f"Download of {s3_key}", exc)
        return False


def list_s3_models(
    bucket_name: str | None,
    prefix: str = "models/",
) -> list[dict[str, Any]]:
    """List model objects under an S3 prefix, newest first."""
    if not is_s3_configured(bucket_name):
        return []

    try:
        normalized_prefix = _normalize_s3_key(prefix)
        if prefix.endswith("/") and not normalized_prefix.endswith("/"):
            normalized_prefix += "/"

        client = _create_s3_client()
        if client is None:
            return []

        models: list[dict[str, Any]] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=str(bucket_name),
            Prefix=normalized_prefix,
        ):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                filename = PurePosixPath(key).name.lower()
                if (
                    Path(key).suffix.lower() not in MODEL_SUFFIXES
                    or "preprocessor" in filename
                ):
                    continue
                models.append(
                    {
                        "key": key,
                        "size": int(item.get("Size", 0)),
                        "last_modified": item.get("LastModified"),
                        "etag": str(item.get("ETag", "")).strip('"'),
                    }
                )

        return sorted(
            models,
            key=lambda item: (
                item["last_modified"].timestamp()
                if hasattr(item["last_modified"], "timestamp")
                else 0.0
            ),
            reverse=True,
        )
    except (BotoCoreError, ClientError, ValueError) as exc:
        _print_aws_error(f"Listing s3://{bucket_name}/{prefix}", exc)
        return []


def get_latest_model_from_s3(
    bucket_name: str | None,
    prefix: str = "models/",
) -> dict[str, Any] | None:
    """Return metadata for the most recently modified model in S3."""
    models = list_s3_models(bucket_name, prefix=prefix)
    return models[0] if models else None
