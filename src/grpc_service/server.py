"""Сервер gRPC Compute Service для распределённой системы MDE-Bench (Лабораторная №3)."""

import os
import signal
import sys
import time
from concurrent import futures
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import grpc
from src.grpc_gen import benchmark_pb2_grpc
from src.grpc_service.runner import detect_device
from src.grpc_service.service import BenchmarkComputeServicer


def serve(host: str = "0.0.0.0", port: int = 50051):
    """Запуск gRPC сервера."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ],
    )

    servicer = BenchmarkComputeServicer(root_dir=root_dir)
    benchmark_pb2_grpc.add_BenchmarkComputeServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    server.start()

    device = detect_device()
    print("=" * 65, flush=True)
    print(f"[gRPC Compute] Server started on {listen_addr}", flush=True)
    print(f"[gRPC Compute] Hardware accelerator: {device}", flush=True)
    print(f"[gRPC Compute] Root directory: {root_dir}", flush=True)
    print("=" * 65, flush=True)

    def handle_shutdown(signum, frame):
        print("\n[gRPC Compute] Stopping server...", flush=True)
        server.stop(grace=3)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1)


if __name__ == "__main__":
    port_env = int(os.environ.get("GRPC_PORT", "50051"))
    host_env = os.environ.get("GRPC_HOST", "0.0.0.0")
    serve(host=host_env, port=port_env)
