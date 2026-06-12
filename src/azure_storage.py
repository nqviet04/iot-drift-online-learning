"""Azure Blob Storage helpers for models and experiment artifacts.

The storage connection string is read from the environment and is never
hard-coded or included in console output.
"""

import os
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from azure.core.exceptions import AzureError
    from azure.storage.blob import BlobServiceClient
except ImportError:  # Keep local experiments usable without Azure dependencies.
    BlobServiceClient = None

    class AzureError(Exception):
        """Fallback exception type used when Azure SDK is unavailable."""


MODEL_SUFFIXES = {".joblib", ".pkl", ".h5", ".keras"}


def _warn(message: str) -> None:
    """Print a friendly Azure warning without exposing secrets."""
    print(f"[AZURE WARNING] {message}")


def _normalize_blob_name(blob_name: str) -> str:
    """Return a portable blob name without a leading slash."""
    normalized = str(
        PurePosixPath(str(blob_name).replace("\\", "/"))
    ).lstrip("/")
    if normalized in {"", "."}:
        raise ValueError("blob_name cannot be empty.")
    return normalized


def is_azure_configured(
    container_name: str | None = None,
    *,
    warn: bool = True,
) -> bool:
    """Check Azure configuration without printing the connection string."""
    missing: list[str] = []
    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        missing.append("AZURE_STORAGE_CONNECTION_STRING")
    if not container_name:
        missing.append("AZURE_BLOB_CONTAINER_NAME")
    if BlobServiceClient is None:
        missing.append("azure-storage-blob")

    if missing and warn:
        _warn(
            "Blob operation skipped because configuration is missing: "
            f"{', '.join(missing)}."
        )
    return not missing


def _create_blob_service_client():
    """Create a BlobServiceClient from the environment connection string."""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if BlobServiceClient is None or not connection_string:
        return None
    return BlobServiceClient.from_connection_string(connection_string)


def _print_azure_error(action: str, error: Exception) -> None:
    """Print a sanitized Azure error without its potentially sensitive text."""
    _warn(
        f"{action} failed ({type(error).__name__}). "
        "Check the container, connection string, network, and permissions."
    )


def upload_file_to_azure_blob(
    local_path: str | Path,
    container_name: str | None,
    blob_name: str,
) -> bool:
    """Upload one local file to Azure Blob Storage."""
    source = Path(local_path)
    if not source.exists() or not source.is_file():
        _warn(f"Upload skipped because local file does not exist: {source}")
        return False
    if not is_azure_configured(container_name):
        return False

    try:
        normalized_name = _normalize_blob_name(blob_name)
        service_client = _create_blob_service_client()
        if service_client is None:
            return False
        blob_client = service_client.get_blob_client(
            container=str(container_name),
            blob=normalized_name,
        )
        with source.open("rb") as file_data:
            blob_client.upload_blob(file_data, overwrite=True)
        return True
    except (AzureError, OSError, ValueError) as exc:
        _print_azure_error(f"Upload of {source.name}", exc)
        return False


def download_file_from_azure_blob(
    container_name: str | None,
    blob_name: str,
    local_path: str | Path,
) -> bool:
    """Download one Azure blob to a local file."""
    if not is_azure_configured(container_name):
        return False

    destination = Path(local_path)
    try:
        normalized_name = _normalize_blob_name(blob_name)
        service_client = _create_blob_service_client()
        if service_client is None:
            return False
        blob_client = service_client.get_blob_client(
            container=str(container_name),
            blob=normalized_name,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob_client.download_blob().readall())
        return True
    except (AzureError, OSError, ValueError) as exc:
        _print_azure_error(f"Download of {blob_name}", exc)
        return False


def list_azure_blobs(
    container_name: str | None,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """List blob metadata under an optional prefix, newest first."""
    if not is_azure_configured(container_name):
        return []

    try:
        service_client = _create_blob_service_client()
        if service_client is None:
            return []
        container_client = service_client.get_container_client(
            str(container_name)
        )
        normalized_prefix = (
            _normalize_blob_name(prefix) if prefix else None
        )
        if (
            prefix
            and prefix.endswith("/")
            and normalized_prefix
            and not normalized_prefix.endswith("/")
        ):
            normalized_prefix += "/"
        blobs = [
            {
                "name": str(blob.name),
                "size": int(getattr(blob, "size", 0) or 0),
                "last_modified": getattr(blob, "last_modified", None),
                "etag": str(getattr(blob, "etag", "") or "").strip('"'),
                "content_type": getattr(
                    getattr(blob, "content_settings", None),
                    "content_type",
                    None,
                ),
            }
            for blob in container_client.list_blobs(
                name_starts_with=normalized_prefix
            )
        ]
        return sorted(
            blobs,
            key=lambda item: (
                item["last_modified"].timestamp()
                if hasattr(item["last_modified"], "timestamp")
                else 0.0
            ),
            reverse=True,
        )
    except (AzureError, ValueError) as exc:
        _print_azure_error(
            f"Listing container '{container_name}'",
            exc,
        )
        return []


def list_azure_models(
    container_name: str | None,
    prefix: str = "models/",
) -> list[dict[str, Any]]:
    """List model artifacts under an Azure blob prefix."""
    return [
        blob
        for blob in list_azure_blobs(container_name, prefix=prefix)
        if (
            Path(blob["name"]).suffix.lower() in MODEL_SUFFIXES
            and "preprocessor" not in PurePosixPath(blob["name"]).name.lower()
        )
    ]


def get_latest_azure_model(
    container_name: str | None,
    prefix: str = "models/",
) -> dict[str, Any] | None:
    """Return metadata for the most recently modified Azure model blob."""
    models = list_azure_models(container_name, prefix=prefix)
    return models[0] if models else None
