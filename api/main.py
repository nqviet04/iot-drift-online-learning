"""FastAPI entrypoint for serving model predictions in future iterations."""

from fastapi import FastAPI


app = FastAPI(title="IoT Drift Online Learning API")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic service health check."""
    return {"status": "ok"}
