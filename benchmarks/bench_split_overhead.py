"""Benchmark: measure split inference overhead components.

Measures:
1. DP noise injection latency
2. HE-LoRA PCMM latency (per layer)
3. Client-side decryption + assembly latency
4. GateLink gate evaluation latency
5. Total split overhead vs simulated base model
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.client.decrypt import CKKSDecryptAssembler
from src.client.dp_noise import DPNoiseInjector
from src.common.types import EncryptedDelta, GateLinkSignal, SplitForwardResponse
from src.server.parallel_helora import (
    CKKSEngine,
    EncryptedAdapter,
    ParallelHELoRAExecutor,
)


def bench_dp_noise(hidden_dim: int = 4096, seq_len: int = 32, n_iter: int = 100) -> float:
    """Benchmark DP noise injection latency."""
    dp = DPNoiseInjector(epsilon=4.0, delta=1e-5, sensitivity=10.0)
    h = torch.randn(1, seq_len, hidden_dim)

    # Warmup
    for _ in range(10):
        dp.inject_noise(h)

    start = time.perf_counter()
    for _ in range(n_iter):
        dp.inject_noise(h)
    elapsed = (time.perf_counter() - start) / n_iter * 1e6  # microseconds

    return elapsed


def bench_pcmm(
    hidden_dim: int = 4096, rank: int = 8, n_layers: int = 32, n_iter: int = 50
) -> float:
    """Benchmark HE-LoRA PCMM per layer."""
    ckks = CKKSEngine()
    executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=True)

    encrypted_B = {}
    plaintext_A = {}
    gate_A = {}

    for i in range(n_layers):
        B = np.random.randn(rank, hidden_dim).astype(np.float64)
        encrypted_B[i] = ckks.encrypt_matrix(B)
        plaintext_A[i] = np.random.randn(hidden_dim, rank).astype(np.float64)
        gate_A[i] = np.random.randn(rank, rank).astype(np.float64)

    adapter = EncryptedAdapter(
        adapter_id="bench",
        encrypted_B=encrypted_B,
        plaintext_A=plaintext_A,
        gate_A=gate_A,
        lora_rank=rank,
        lora_alpha=16.0,
        num_layers=n_layers,
    )
    executor.register_adapter(adapter)

    h = np.random.randn(hidden_dim).astype(np.float64)

    # Warmup
    for _ in range(5):
        executor.compute_layer_delta(adapter, 0, h)

    # Benchmark single layer
    start = time.perf_counter()
    for _ in range(n_iter):
        executor.compute_layer_delta(adapter, 0, h)
    per_layer_us = (time.perf_counter() - start) / n_iter * 1e6

    return per_layer_us


def bench_all_layers(
    hidden_dim: int = 4096, rank: int = 8, n_layers: int = 32, n_iter: int = 20
) -> float:
    """Benchmark all-layer HE-LoRA computation."""
    ckks = CKKSEngine()
    executor = ParallelHELoRAExecutor(
        ckks_engine=ckks, gatelink_enabled=True, max_parallel_layers=4
    )

    encrypted_B = {}
    plaintext_A = {}
    gate_A = {}

    for i in range(n_layers):
        B = np.random.randn(rank, hidden_dim).astype(np.float64)
        encrypted_B[i] = ckks.encrypt_matrix(B)
        plaintext_A[i] = np.random.randn(hidden_dim, rank).astype(np.float64)
        gate_A[i] = np.random.randn(rank, rank).astype(np.float64)

    adapter = EncryptedAdapter(
        adapter_id="bench",
        encrypted_B=encrypted_B,
        plaintext_A=plaintext_A,
        gate_A=gate_A,
        lora_rank=rank,
        lora_alpha=16.0,
        num_layers=n_layers,
    )
    executor.register_adapter(adapter)

    layer_states = {i: np.random.randn(hidden_dim).astype(np.float64) for i in range(n_layers)}

    # Warmup
    for _ in range(3):
        executor.compute_all_deltas("bench", layer_states)

    start = time.perf_counter()
    for _ in range(n_iter):
        executor.compute_all_deltas("bench", layer_states)
    total_ms = (time.perf_counter() - start) / n_iter * 1e3

    return total_ms


def bench_client_assembly(hidden_dim: int = 4096, n_layers: int = 32, n_iter: int = 50) -> float:
    """Benchmark client-side decryption + assembly."""
    assembler = CKKSDecryptAssembler()
    ckks = CKKSEngine()

    # Create mock response
    base = np.random.randn(1, hidden_dim).astype(np.float64)

    import struct
    deltas = []
    signals = []
    for i in range(n_layers):
        d = np.random.randn(hidden_dim).astype(np.float64)
        header = struct.pack("!II", hidden_dim, 1)
        deltas.append(EncryptedDelta(
            layer_idx=i,
            ciphertext_bytes=header + d.tobytes(),
            num_elements=hidden_dim,
        ))

        g = np.random.randn(8).astype(np.float64)
        g_header = struct.pack("!II", 8, 1)
        signals.append(GateLinkSignal(
            layer_idx=i,
            ciphertext_bytes=g_header + g.tobytes(),
            gate_rank=8,
            activation_fn="silu",
        ))

    response = SplitForwardResponse(
        base_hidden_states=base,
        encrypted_deltas=deltas,
        gatelink_signals=signals,
        layers_computed=n_layers,
    )

    # Warmup
    for _ in range(5):
        assembler.assemble(response)

    start = time.perf_counter()
    for _ in range(n_iter):
        assembler.assemble(response)
    elapsed_ms = (time.perf_counter() - start) / n_iter * 1e3

    return elapsed_ms


def main():
    print("=" * 60)
    print("Split Inference Overhead Benchmark")
    print("=" * 60)

    configs = [
        {"hidden_dim": 2048, "rank": 8, "n_layers": 16, "label": "Llama-3.2-1B (16 layers, r=8)"},
        {"hidden_dim": 4096, "rank": 8, "n_layers": 32, "label": "Llama-3-8B (32 layers, r=8)"},
        {"hidden_dim": 4096, "rank": 32, "n_layers": 32, "label": "Llama-3-8B (32 layers, r=32)"},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")

        dp_us = bench_dp_noise(hidden_dim=cfg["hidden_dim"])
        print(f"  DP noise injection:    {dp_us:.0f} μs")

        pcmm_us = bench_pcmm(hidden_dim=cfg["hidden_dim"], rank=cfg["rank"], n_layers=cfg["n_layers"])
        print(f"  PCMM per layer:        {pcmm_us:.0f} μs")

        all_ms = bench_all_layers(
            hidden_dim=cfg["hidden_dim"], rank=cfg["rank"], n_layers=cfg["n_layers"]
        )
        print(f"  All layers (parallel): {all_ms:.1f} ms")

        assembly_ms = bench_client_assembly(
            hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"]
        )
        print(f"  Client assembly:       {assembly_ms:.1f} ms")

        total_ms = dp_us / 1000 + all_ms + assembly_ms
        print(f"  Total split overhead:  {total_ms:.1f} ms/token")

        base_ms = 1000.0 / 53.18  # vLLM baseline
        overhead_pct = (total_ms / base_ms) * 100
        print(f"  Overhead vs vLLM:      {overhead_pct:.0f}%")


if __name__ == "__main__":
    main()
