"""Upload local models, metrics, and figures to the configured AWS S3 bucket."""

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.aws_storage import is_s3_configured, upload_file_to_s3
from src.config import (
    AWS_S3_BUCKET_NAME,
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
    """Collect local artifacts and their destination S3 keys."""
    candidates: list[tuple[Path, str]] = []
    for local_dir, s3_prefix in UPLOAD_GROUPS:
        if not local_dir.exists():
            continue
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            relative_key = path.relative_to(local_dir).as_posix()
            candidates.append((path, f"{s3_prefix}/{relative_key}"))
    return candidates


def main() -> None:
    """Upload all current experiment artifacts without exposing credentials."""
    if not is_s3_configured(AWS_S3_BUCKET_NAME):
        print("No files were uploaded. Configure AWS variables and run again.")
        return

    candidates = _upload_candidates()
    if not candidates:
        print("No local model, metric, or figure files were found to upload.")
        return

    uploaded = 0
    failed = 0
    for local_path, s3_key in candidates:
        print(
            f"[UPLOAD] {_relative(local_path)} "
            f"-> s3://{AWS_S3_BUCKET_NAME}/{s3_key}"
        )
        if upload_file_to_s3(local_path, AWS_S3_BUCKET_NAME, s3_key):
            uploaded += 1
            print(f"[OK] {s3_key}")
        else:
            failed += 1
            print(f"[FAILED] {s3_key}")

    print("\nS3 upload summary")
    print(f"Bucket: {AWS_S3_BUCKET_NAME}")
    print(f"Uploaded: {uploaded}")
    print(f"Failed: {failed}")
    print(f"Total: {len(candidates)}")


if __name__ == "__main__":
    main()
