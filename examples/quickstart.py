"""Quickstart: Split Inference with Three-Layer Privacy.

Demonstrates the full split inference pipeline with simulated components
(no real LLM required). Shows:
1. Device-adaptive compilation
2. DP noise injection (input privacy)
3. Parallel HE-LoRA computation (adapter privacy)
4. GateLink gate evaluation (zero approximation error)
5. Client-side assembly (output privacy)

Usage:
    python -m examples.quickstart
"""

from __future__ import annotations

import numpy as np
import struct
import torch

from src.client.decrypt import CKKSDecryptAssembler
from src.client.dp_noise import DPNoiseInjector
from src.common.config import SplitInferenceConfig, PrivacyConfig
from src.common.types import (
    EncryptedDelta,
    GateLinkSignal,
    SplitForwardResponse,
)
from src.compiler.device_profiles import get_profile
from src.compiler.privacy_budget import PrivacyBudgetOptimizer
from src.compiler.split_compiler import SplitCompiler
from src.server.parallel_helora import (
    CKKSEngine,
    EncryptedAdapter,
    ParallelHELoRAExecutor,
)


def main():
    print("=" * 60)
    print("Split Inference — Quickstart Demo")
    print("=" * 60)

    # === Step 1: Compile split schedule for device ===
    print("\n[1] Compiling split schedule for 'laptop' profile...")
    compiler = SplitCompiler(
        model_id="demo-model",
        total_layers=8,
        hidden_size=128,
    )
    report = compiler.compile("laptop")

    print(f"    Model: {report.config.model_id}")
    print(f"    Client layers (K): {report.config.num_client_layers}")
    print(f"    Server layers: {report.config.num_server_layers}")
    print(f"    DP epsilon: {report.config.privacy.epsilon}")
    print(f"    LoRA rank: {report.config.lora_rank}")
    print(f"    Estimated throughput: {report.budget.estimated_throughput_tps:.1f} tok/s")
    print(f"    Privacy score: {report.budget.privacy_score:.2f}")
    if report.warnings:
        for w in report.warnings:
            print(f"    Warning: {w}")

    # === Step 2: Simulate client-side processing ===
    hidden_dim = 128
    seq_len = 5
    rank = report.config.lora_rank

    print(f"\n[2] Client: embedding + {report.config.num_client_layers} layers (simulated)...")
    h_K = torch.randn(1, seq_len, hidden_dim)  # Simulated post-K-layer hidden states
    print(f"    Hidden states shape: {h_K.shape}")
    print(f"    Hidden states norm: {torch.norm(h_K).item():.2f}")

    # === Step 3: DP noise injection ===
    print(f"\n[3] Injecting DP noise (epsilon={report.config.privacy.epsilon})...")
    dp = DPNoiseInjector(
        epsilon=report.config.privacy.epsilon,
        delta=1e-5,
        sensitivity=5.0,
    )
    h_noised, stats = dp.inject_noise(h_K)

    print(f"    Noise sigma: {stats.sigma:.4f}")
    print(f"    Signal norm: {stats.signal_norm:.2f}")
    print(f"    Noise norm: {stats.noise_norm:.2f}")
    print(f"    SNR: {stats.snr_db:.1f} dB")
    print(f"    Privacy: {DPNoiseInjector.privacy_guarantee_summary(dp.epsilon, dp.delta)}")

    # === Step 4: Server-side HE-LoRA computation ===
    num_server_layers = report.config.num_server_layers
    K = report.config.num_client_layers

    print(f"\n[4] Server: base model ({num_server_layers} layers) + parallel HE-LoRA...")

    # Simulate base model output
    h_base = h_noised.squeeze(0).numpy() * 0.95  # Simulated transformation

    # Set up HE-LoRA executor
    ckks = CKKSEngine()
    executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=True)

    encrypted_B = {}
    plaintext_A = {}
    gate_A = {}
    for i in range(num_server_layers):
        layer_idx = K + i
        B = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
        encrypted_B[layer_idx] = ckks.encrypt_matrix(B)
        plaintext_A[layer_idx] = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01
        gate_A[layer_idx] = np.random.randn(rank, rank).astype(np.float64) * 0.01

    adapter = EncryptedAdapter(
        adapter_id="default",
        encrypted_B=encrypted_B,
        plaintext_A=plaintext_A,
        gate_A=gate_A,
        lora_rank=rank,
        lora_alpha=16.0,
        num_layers=num_server_layers,
    )
    executor.register_adapter(adapter)

    layer_states = {
        K + i: h_base[0].copy()  # Each layer sees base path hidden states
        for i in range(num_server_layers)
    }
    deltas, signals = executor.compute_all_deltas("default", layer_states)

    print(f"    Base output shape: {h_base.shape}")
    print(f"    Encrypted deltas: {len(deltas)} layers")
    print(f"    GateLink signals: {len(signals)} layers")
    print(f"    Round trips: 1 (fused protocol)")

    # === Step 5: Client assembly ===
    print("\n[5] Client: decrypt + GateLink gates + assembly...")

    response = SplitForwardResponse(
        base_hidden_states=h_base,
        encrypted_deltas=deltas,
        gatelink_signals=signals,
        layers_computed=num_server_layers,
    )

    assembler = CKKSDecryptAssembler()
    h_final = assembler.assemble(response)

    print(f"    Final hidden states shape: {h_final.shape}")
    print(f"    Final norm: {torch.norm(h_final).item():.2f}")

    # === Step 6: Client LM head + sampling (simulated) ===
    print("\n[6] Client: LM head + sampling (simulated)...")
    logits = torch.randn(1, 100)  # Simulated vocab logits
    probs = torch.softmax(logits / 0.7, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    print(f"    Sampled token ID: {next_token}")
    print(f"    (Server never saw logits or token — output privacy preserved)")

    # === Summary ===
    print("\n" + "=" * 60)
    print("Privacy Summary")
    print("=" * 60)
    print(f"  Input privacy:   epsilon={dp.epsilon} DP noise (server sees only noised states)")
    print(f"  Adapter privacy: CKKS-encrypted B matrices (ZeRo-MOAI, zero rotations)")
    print(f"  Output privacy:  LM head + sampling on client (logits never transmitted)")
    print(f"  GateLink:        Exact non-linear gates (zero approximation error)")
    print(f"  Protocol:        1 round trip per token (GateLink-fused)")
    print(f"  Device profile:  {report.profile.name} (K={K}, rank={rank})")


if __name__ == "__main__":
    main()
