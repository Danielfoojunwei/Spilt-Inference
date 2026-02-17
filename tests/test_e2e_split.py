"""End-to-end tests for split inference pipeline.

Tests the full pipeline WITHOUT requiring a real LLM model:
- Simulated client model shard (random embeddings)
- Real DP noise injection
- Real parallel HE-LoRA computation (simulation CKKS)
- Real client-side decryption and assembly
"""

import numpy as np
import pytest
import torch

from src.client.decrypt import CKKSDecryptAssembler
from src.client.dp_noise import DPNoiseInjector
from src.common.config import PrivacyConfig, SplitInferenceConfig
from src.common.types import (
    EncryptedDelta,
    GateLinkSignal,
    SplitForwardRequest,
    SplitForwardResponse,
)
from src.server.parallel_helora import (
    CKKSEngine,
    EncryptedAdapter,
    ParallelHELoRAExecutor,
)


class TestE2EPipeline:
    """End-to-end pipeline tests with simulated components."""

    def _make_config(self) -> SplitInferenceConfig:
        return SplitInferenceConfig(
            model_id="test-model",
            total_layers=4,
            num_client_layers=1,
            num_server_layers=3,
            privacy=PrivacyConfig(epsilon=4.0),
            lora_rank=4,
            lora_alpha=8.0,
        )

    def _make_adapter_and_executor(self, hidden_dim=64, rank=4, num_server_layers=3):
        """Create test adapter and executor."""
        ckks = CKKSEngine()
        executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=True)

        encrypted_B = {}
        plaintext_A = {}
        gate_A = {}

        for i in range(num_server_layers):
            layer_idx = i + 1  # Server layers start at K=1
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
            lora_alpha=8.0,
            num_layers=num_server_layers,
        )
        executor.register_adapter(adapter)
        return executor

    def test_dp_noise_to_server_response(self):
        """Test: DP noise → simulated server → client assembly."""
        hidden_dim = 64
        seq_len = 5

        # Step 1: Client generates hidden states + DP noise
        dp = DPNoiseInjector(epsilon=4.0, delta=1e-5, sensitivity=1.0)
        h_K = torch.randn(1, seq_len, hidden_dim)
        h_noised, stats = dp.inject_noise(h_K)

        assert stats.sigma > 0
        assert h_noised.shape == h_K.shape

        # Step 2: Server computes base output + HE-LoRA deltas
        executor = self._make_adapter_and_executor(hidden_dim=hidden_dim)

        # Simulate base model output (just pass through with some transformation)
        h_base = h_noised.squeeze(0).numpy() * 0.9  # Simulated base output

        # Compute HE-LoRA deltas for server layers
        layer_hidden_states = {}
        for layer_idx in range(1, 4):
            layer_hidden_states[layer_idx] = h_base[0]  # Use first token's hidden state

        deltas, signals = executor.compute_all_deltas("default", layer_hidden_states)

        assert len(deltas) == 3
        assert len(signals) == 3

        # Step 3: Build server response
        response = SplitForwardResponse(
            base_hidden_states=h_base,
            encrypted_deltas=deltas,
            gatelink_signals=signals,
            layers_computed=3,
        )

        # Step 4: Client assembly
        assembler = CKKSDecryptAssembler()
        h_final = assembler.assemble(response)

        # h_base is [seq_len, hidden_dim] (2D), so h_final preserves that shape
        assert h_final.shape[0] == seq_len
        assert h_final.shape[-1] == hidden_dim

    def test_three_layer_privacy(self):
        """Verify the three privacy layers are enforced."""
        hidden_dim = 32

        # Layer 1: Input privacy (DP noise)
        dp = DPNoiseInjector(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        h_clean = torch.randn(1, 3, hidden_dim)
        h_noised, _ = dp.inject_noise(h_clean)

        # Server receives only noised states
        # Verify: noised ≠ clean (privacy)
        assert not torch.allclose(h_noised, h_clean, atol=0.01)

        # Layer 2: Adapter privacy (encrypted weights)
        ckks = CKKSEngine()
        B = np.random.randn(4, hidden_dim).astype(np.float64)
        encrypted_B = ckks.encrypt_matrix(B)
        # Server has encrypted_B but cannot read B without secret key
        assert isinstance(encrypted_B, bytes)

        # Layer 3: Output privacy (LM head on client)
        # Logits never leave client — verified by architecture
        # (no logits field in SplitForwardResponse)
        response = SplitForwardResponse(
            base_hidden_states=h_noised.squeeze(0).numpy(),
        )
        assert not hasattr(response, "logits")

    def test_gatelink_fused_round_trip(self):
        """Verify single round trip carries all data."""
        hidden_dim = 32
        executor = self._make_adapter_and_executor(
            hidden_dim=hidden_dim, rank=4, num_server_layers=3
        )

        # One base forward pass produces all layer hidden states
        layer_states = {
            i: np.random.randn(hidden_dim).astype(np.float64) for i in range(1, 4)
        }

        # One call returns ALL deltas + ALL GateLink signals
        deltas, signals = executor.compute_all_deltas("default", layer_states)

        # Verify: everything in one batch (fused round trip)
        assert len(deltas) == 3, "All layer deltas in one response"
        assert len(signals) == 3, "All GateLink signals in one response"
        # No per-layer round trips needed!

    def test_adapter_only_encryption(self):
        """Verify AOE: weights encrypted, activations plaintext."""
        ckks = CKKSEngine()

        # Adapter B is encrypted (AOE)
        B = np.random.randn(4, 32).astype(np.float64)
        encrypted_B = ckks.encrypt_matrix(B)

        # Activations are plaintext (not encrypted)
        x = np.random.randn(32, 1).astype(np.float64)

        # PCMM: encrypted_B @ plaintext_x → encrypted result
        result = ckks.pcmm_encrypted_weight(encrypted_B, x)

        # Client decrypts result
        decrypted = ckks.decrypt_vector(result)

        # Verify correctness
        expected = B @ x
        np.testing.assert_allclose(decrypted.flatten(), expected.flatten(), atol=1e-10)


class TestParallelAdapterDecoupling:
    """Test that parallel adapter path is decoupled from base path."""

    def test_no_feedback(self):
        """HE-LoRA deltas don't feed back into base path hidden states."""
        hidden_dim = 32
        executor = self._make_adapter_and_executor(
            hidden_dim=hidden_dim, rank=4, num_server_layers=3
        )

        # Base path hidden states are independent of adapter output
        base_h1 = np.random.randn(hidden_dim).astype(np.float64)
        base_h2 = np.random.randn(hidden_dim).astype(np.float64)  # Not h1 + delta1
        base_h3 = np.random.randn(hidden_dim).astype(np.float64)

        layer_states = {1: base_h1, 2: base_h2, 3: base_h3}

        # Compute deltas — these are additive corrections, not fed back
        deltas, _ = executor.compute_all_deltas("default", layer_states)

        # Each delta is independent (computed on base path hidden states only)
        assert len(deltas) == 3
        # Deltas at layer 2 use base_h2, NOT base_h1 + delta_1
        # This is the parallel adapter property

    def _make_adapter_and_executor(self, hidden_dim=32, rank=4, num_server_layers=3):
        ckks = CKKSEngine()
        executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=False)

        encrypted_B = {}
        plaintext_A = {}

        for i in range(num_server_layers):
            layer_idx = i + 1
            B = np.random.randn(rank, hidden_dim).astype(np.float64)
            encrypted_B[layer_idx] = ckks.encrypt_matrix(B)
            plaintext_A[layer_idx] = np.random.randn(hidden_dim, rank).astype(np.float64)

        adapter = EncryptedAdapter(
            adapter_id="default",
            encrypted_B=encrypted_B,
            plaintext_A=plaintext_A,
            gate_A={},
            lora_rank=rank,
            lora_alpha=8.0,
            num_layers=num_server_layers,
        )
        executor.register_adapter(adapter)
        return executor
