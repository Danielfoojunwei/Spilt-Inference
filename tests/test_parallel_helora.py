"""Tests for parallel HE-LoRA executor."""

import numpy as np
import pytest

from src.server.parallel_helora import (
    CKKSEngine,
    EncryptedAdapter,
    ParallelHELoRAExecutor,
)


def _make_adapter(num_layers: int = 4, rank: int = 8, hidden_dim: int = 64) -> EncryptedAdapter:
    """Create a test adapter with random weights."""
    ckks = CKKSEngine()

    encrypted_B = {}
    plaintext_A = {}
    gate_A = {}

    for i in range(num_layers):
        # B: [rank, hidden_dim] — encrypted
        B = np.random.randn(rank, hidden_dim).astype(np.float64)
        encrypted_B[i] = ckks.encrypt_matrix(B)

        # A: [hidden_dim, rank] — plaintext (FFA-LoRA: freeze-A)
        plaintext_A[i] = np.random.randn(hidden_dim, rank).astype(np.float64)

        # Gate A: [rank, rank] — for GateLink pre-activation
        gate_A[i] = np.random.randn(rank, rank).astype(np.float64)

    return EncryptedAdapter(
        adapter_id="test-adapter",
        encrypted_B=encrypted_B,
        plaintext_A=plaintext_A,
        gate_A=gate_A,
        lora_rank=rank,
        lora_alpha=16.0,
        num_layers=num_layers,
    )


class TestCKKSEngine:
    """Test CKKS simulation engine."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypt then decrypt should recover original data."""
        ckks = CKKSEngine()
        matrix = np.random.randn(8, 64).astype(np.float64)

        encrypted = ckks.encrypt_matrix(matrix)
        decrypted = ckks.decrypt_vector(encrypted)

        np.testing.assert_allclose(decrypted, matrix, atol=1e-10)

    def test_pcmm_correctness(self):
        """PCMM should compute correct matrix multiplication."""
        ckks = CKKSEngine()

        weight = np.random.randn(8, 64).astype(np.float64)
        x = np.random.randn(64, 1).astype(np.float64)

        encrypted_weight = ckks.encrypt_matrix(weight)
        result_encrypted = ckks.pcmm_encrypted_weight(encrypted_weight, x)
        result = ckks.decrypt_vector(result_encrypted)

        expected = weight @ x
        np.testing.assert_allclose(result.flatten(), expected.flatten(), atol=1e-10)


class TestParallelHELoRAExecutor:
    """Test parallel encrypted adapter injection."""

    def test_single_layer_delta(self):
        """Compute encrypted delta for one layer."""
        ckks = CKKSEngine()
        executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=True)

        adapter = _make_adapter(num_layers=1, rank=8, hidden_dim=64)
        executor.register_adapter(adapter)

        h = np.random.randn(64).astype(np.float64)
        delta, signal = executor.compute_layer_delta(adapter, 0, h)

        assert delta.layer_idx == 0
        assert len(delta.ciphertext_bytes) > 0
        assert delta.num_elements > 0
        assert signal is not None
        assert signal.layer_idx == 0
        assert signal.activation_fn == "silu"

    def test_all_layers_batch(self):
        """Compute deltas for all layers in one batch."""
        ckks = CKKSEngine()
        executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=True)

        num_layers = 4
        hidden_dim = 64
        adapter = _make_adapter(num_layers=num_layers, rank=8, hidden_dim=hidden_dim)
        executor.register_adapter(adapter)

        # Simulate hidden states from base path
        layer_hidden_states = {
            i: np.random.randn(hidden_dim).astype(np.float64)
            for i in range(num_layers)
        }

        deltas, signals = executor.compute_all_deltas("test-adapter", layer_hidden_states)

        assert len(deltas) == num_layers
        assert len(signals) == num_layers
        for i, delta in enumerate(deltas):
            assert delta.layer_idx == i

    def test_gatelink_disabled(self):
        """When GateLink is disabled, no signals are returned."""
        ckks = CKKSEngine()
        executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=False)

        adapter = _make_adapter(num_layers=2, rank=8, hidden_dim=64)
        executor.register_adapter(adapter)

        layer_hidden_states = {
            i: np.random.randn(64).astype(np.float64) for i in range(2)
        }

        deltas, signals = executor.compute_all_deltas("test-adapter", layer_hidden_states)

        assert len(deltas) == 2
        assert len(signals) == 0  # No GateLink signals

    def test_delta_numerical_correctness(self):
        """Verify delta = (alpha/r) * A @ B @ x."""
        ckks = CKKSEngine()
        executor = ParallelHELoRAExecutor(ckks_engine=ckks, gatelink_enabled=False)

        rank = 4
        hidden_dim = 32
        alpha = 16.0

        # Create adapter with known weights
        B = np.random.randn(rank, hidden_dim).astype(np.float64)
        A = np.random.randn(hidden_dim, rank).astype(np.float64)

        adapter = EncryptedAdapter(
            adapter_id="numerical-test",
            encrypted_B={0: ckks.encrypt_matrix(B)},
            plaintext_A={0: A},
            gate_A={},
            lora_rank=rank,
            lora_alpha=alpha,
            num_layers=1,
        )
        executor.register_adapter(adapter)

        x = np.random.randn(hidden_dim).astype(np.float64)

        delta, _ = executor.compute_layer_delta(adapter, 0, x)

        # Decrypt and verify
        import struct
        header_size = struct.calcsize("!II")
        rows, cols = struct.unpack("!II", delta.ciphertext_bytes[:header_size])
        result = np.frombuffer(delta.ciphertext_bytes[header_size:], dtype=np.float64)

        expected = (alpha / rank) * A @ B @ x
        np.testing.assert_allclose(result[:hidden_dim], expected, atol=1e-8)

    def test_missing_adapter_raises(self):
        """Requesting unknown adapter raises error."""
        ckks = CKKSEngine()
        executor = ParallelHELoRAExecutor(ckks_engine=ckks)

        with pytest.raises(KeyError):
            executor.compute_all_deltas("nonexistent", {0: np.zeros(64)})
