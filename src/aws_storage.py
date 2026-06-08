"""AWS S3 storage extension points.

This module intentionally does not read hard-coded credentials. Use environment
variables or IAM roles when S3 upload/download is implemented.
"""


def upload_model_to_s3(*args, **kwargs):
    """Placeholder for future S3 model upload support."""
    raise NotImplementedError("S3 upload is planned for a future extension.")
