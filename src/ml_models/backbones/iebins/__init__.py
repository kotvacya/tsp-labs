"""Архитектура IEBins (Iterative Elastic Bins, CVPR 2024)."""

from .networks.NewCRFDepth import NewCRFDepth
from . import utils

__all__ = ["NewCRFDepth", "utils"]
