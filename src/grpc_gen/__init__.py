import sys
from pathlib import Path

_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from . import benchmark_pb2, benchmark_pb2_grpc

__all__ = ["benchmark_pb2", "benchmark_pb2_grpc"]
