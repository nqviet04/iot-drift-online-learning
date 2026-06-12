"""Upload local models, metrics, and figures to Azure Blob Storage."""

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.azure_storage import (
    is_azure_configured,
    upload_file_to_azure_blob,
)
from src.config import (
    AZURE_BLOB_CONTAINER_NAME,
    CLOUD_MODEL_DIR,
    FIGURE_DIR,
    METRIC_DIR,
)


UPLOAD_GROUPS = (
    (CLOUD_MODEL_DIR, "models"),
    (METRIC_DIR, "metrics"),
    (FIGURE_DIR, "figures"),
)


def _relative(path: Path) -> Path:
    """Return a readable project-relative path."""
    return path.resolve().relative_to(ROOT_DIR.resolve())


def _upload_candidates() -> list[tuple[Path, str]]:
    """Collect local artifacts and their destination blob names."""
    candidates: list[tuple[Path, str]] = []
    for local_dir, blob_prefix in UPLOAD_GROUPS:
        if not local_dir.exists():
            continue
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            relative_name = path.relative_to(local_dir).as_posix()
            candidates.append(
                (path, f"{blob_prefix}/{relative_name}")
            )
    return candidates


def main() -> None:
    """Upload all current artifacts without exposing Azure secrets."""
    if not is_azure_configured(AZURE_BLOB_CONTAINER_NAME):
        print(
            "No files were uploaded. Configure Azure variables and run again."
        )
        return

    candidates = _upload_candidates()
    if not candidates:
        print("No local model, metric, or figure files were found to upload.")
        return

    uploaded = 0
    failed = 0
    for local_path, blob_name in candidates:
        print(
            f"[UPLOAD] {_relative(local_path)} "
            f"-> {AZURE_BLOB_CONTAINER_NAME}/{blob_name}"
        )
        if upload_file_to_azure_blob(
            local_path,
            AZURE_BLOB_CONTAINER_NAME,
            blob_name,
        ):
            uploaded += 1
            print(f"[OK] {blob_name}")
        else:
            failed += 1
            print(f"[FAILED] {blob_name}")

    print("\nAzure Blob upload summary")
    print(f"Container: {AZURE_BLOB_CONTAINER_NAME}")
    print(f"Uploaded: {uploaded}")
    print(f"Failed: {failed}")
    print(f"Total: {len(candidates)}")


if __name__ == "__main__":
    main()
