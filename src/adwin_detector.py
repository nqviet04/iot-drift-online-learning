"""ADWIN drift detector wrapper."""

from river.drift import ADWIN


def build_adwin(delta: float = 0.002) -> ADWIN:
    """Create an ADWIN detector instance."""
    return ADWIN(delta=delta)
