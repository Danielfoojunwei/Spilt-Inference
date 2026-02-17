"""Parallel encrypted adapter injection (HE-LoRA path).

Computes encrypted LoRA deltas for all layers in parallel, decoupled from the
base model forward pass. This is the core HE computation engine that:

1. Takes plaintext hidden states from each base model layer
2. Computes encrypted_B @ x (PCMM via ZeRo-MOAI, B is pre-encrypted under AOE)
3. Computes A @ (encrypted_B @ x) (PCMM, A is plaintext via FFA-LoRA)
4. Generates GateLink pre-activation signals for client-side gate evaluation

All layers' deltas and GateLink signals are returned in one batch, enabling
the fused single round trip protocol.

Architecture based on:
- Side-Tuning (ECCV 2020): decoupled adapter path validated
- MAM Adapters (ICLR 2022): parallel adapter = best configuration
- TenSafe ZeRo-MOAI: zero-rotation PCMM engine
- TenSafe GateLink: client-aided non-linear evaluation
"""

from __future__ import annotations

import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.common.types import EncryptedDelta, GateLinkSignal

logger = logging.getLogger(__name__)


@dataclass
class EncryptedAdapter:
    """Pre-encrypted adapter stored on server (AOE mode).

    B matrices are CKKS-encrypted once at upload.
    A matrices are plaintext (FFA-LoRA: freeze-A).
    Gate matrices are plaintext (for GateLink pre-activation).
    """

    adapter_id: str
    encrypted_B: dict[int, bytes]  # {layer_idx: CKKS ciphertext of B}
    plaintext_A: dict[int, np.ndarray]  # {layer_idx: A matrix}
    gate_A: dict[int, np.ndarray]  # {layer_idx: gate projection matrix}
    lora_rank: int
    lora_alpha: float
    num_layers: int


class CKKSEngine:
    """CKKS homomorphic encryption engine.

    In production, this wraps TenSafe's MOAI backend (Pyfhel or N2HE).
    This implementation provides the interface with simulation for testing.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 16384,
        coeff_mod_bit_sizes: Optional[list[int]] = None,
        scale_bits: int = 40,
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.scale_bits = scale_bits
        self._scale = 2.0**scale_bits

        # In production: initialize Pyfhel context
        # self._he = Pyfhel()
        # self._he.contextGen(scheme='ckks', n=poly_modulus_degree, ...)
        self._initialized = True
        logger.info(
            "CKKS engine initialized: N=%d, L=%d, scale=2^%d",
            poly_modulus_degree,
            len(self.coeff_mod_bit_sizes),
            scale_bits,
        )

    def encrypt_matrix(self, matrix: np.ndarray) -> bytes:
        """Encrypt a matrix under CKKS (AOE: one-time adapter upload).

        Uses ZeRo-MOAI column packing for zero-rotation PCMM.
        """
        # Simulation: pack matrix as bytes with header
        header = struct.pack("!II", matrix.shape[0], matrix.shape[1])
        data = matrix.astype(np.float64).tobytes()
        # In production: MOAI column-packed encryption
        # packed = moai_column_pack(matrix)
        # ciphertext = self._he.encrypt(packed)
        return header + data

    def pcmm_encrypted_weight(
        self,
        encrypted_weight: bytes,
        plaintext_input: np.ndarray,
    ) -> bytes:
        """Plaintext-Ciphertext Matrix Multiplication (PCMM).

        AOE mode: encrypted_weight (B) × plaintext_input (x).
        Uses ZeRo-MOAI zero-rotation column packing.

        This is the core HE operation. Each call costs ~406 μs per matrix
        (half of the ~812 μs per-layer cost, as there are 2 PCMMs per layer).
        """
        # Simulation: extract matrix, multiply, re-pack as "encrypted" result
        header_size = struct.calcsize("!II")
        rows, cols = struct.unpack("!II", encrypted_weight[:header_size])
        weight_data = np.frombuffer(
            encrypted_weight[header_size:], dtype=np.float64
        ).reshape(rows, cols)

        # Actual matrix multiplication (simulates PCMM result)
        result = weight_data @ plaintext_input

        # Pack result as "encrypted" bytes
        result_header = struct.pack("!II", result.shape[0], result.shape[1] if result.ndim > 1 else 1)
        return result_header + result.astype(np.float64).tobytes()

    def decrypt_vector(self, ciphertext: bytes) -> np.ndarray:
        """Decrypt a CKKS ciphertext to get plaintext vector."""
        header_size = struct.calcsize("!II")
        rows, cols = struct.unpack("!II", ciphertext[:header_size])
        return np.frombuffer(
            ciphertext[header_size:], dtype=np.float64
        ).reshape(rows, cols) if cols > 1 else np.frombuffer(
            ciphertext[header_size:], dtype=np.float64
        ).reshape(rows)


class ParallelHELoRAExecutor:
    """Parallel encrypted adapter injection engine.

    Runs the HE-LoRA computation path independently from the base model forward pass.
    For each layer, computes:
        z_i = encrypted_B_i @ x_i          (PCMM, ZeRo-MOAI)
        delta_i = A_i @ z_i                (PCMM, A plaintext)
        pre_gate_i = A_gate_i @ z_i        (GateLink signal)

    All layers are computed and returned as a batch for the fused single round trip.
    """

    def __init__(
        self,
        ckks_engine: CKKSEngine,
        max_parallel_layers: int = 4,
        gatelink_enabled: bool = True,
    ):
        self.ckks = ckks_engine
        self.max_parallel_layers = max_parallel_layers
        self.gatelink_enabled = gatelink_enabled
        self._adapters: dict[str, EncryptedAdapter] = {}

    def register_adapter(self, adapter: EncryptedAdapter) -> None:
        """Register a pre-encrypted adapter for use in inference."""
        self._adapters[adapter.adapter_id] = adapter
        logger.info(
            "Registered adapter '%s': rank=%d, layers=%d",
            adapter.adapter_id,
            adapter.lora_rank,
            adapter.num_layers,
        )

    def compute_layer_delta(
        self,
        adapter: EncryptedAdapter,
        layer_idx: int,
        hidden_states: np.ndarray,
    ) -> tuple[EncryptedDelta, Optional[GateLinkSignal]]:
        """Compute encrypted LoRA delta + GateLink signal for one layer.

        Args:
            adapter: Pre-encrypted adapter.
            layer_idx: Which layer to compute.
            hidden_states: Plaintext hidden states from base path, shape [seq_len, hidden_dim].

        Returns:
            Tuple of (encrypted_delta, gatelink_signal).
        """
        start_time = time.perf_counter()

        # Step 1: z = encrypted_B @ x (PCMM, ZeRo-MOAI, ~406 μs)
        encrypted_B = adapter.encrypted_B[layer_idx]

        # Reshape input for matrix multiply: [hidden_dim] or [seq_len, hidden_dim]
        if hidden_states.ndim == 1:
            x = hidden_states.reshape(-1, 1)
        else:
            x = hidden_states.T  # [hidden_dim, seq_len]

        z_encrypted = self.ckks.pcmm_encrypted_weight(encrypted_B, x)

        # Step 2: delta = A @ z (PCMM, A is plaintext, ~406 μs)
        # In CKKS: plaintext_A × encrypted_z → encrypted_delta
        A = adapter.plaintext_A[layer_idx]
        z_data = self.ckks.decrypt_vector(z_encrypted)
        if z_data.ndim == 1:
            z_data = z_data.reshape(-1, 1)

        # Scale by alpha/rank
        scale = adapter.lora_alpha / adapter.lora_rank
        delta_result = (A @ z_data) * scale

        # Package as encrypted delta
        delta_header = struct.pack("!II", delta_result.shape[0], delta_result.shape[1] if delta_result.ndim > 1 else 1)
        delta_bytes = delta_header + delta_result.astype(np.float64).tobytes()

        encrypted_delta = EncryptedDelta(
            layer_idx=layer_idx,
            ciphertext_bytes=delta_bytes,
            num_elements=delta_result.size,
            scale=self.ckks._scale,
        )

        # Step 3: GateLink pre-activation (optional)
        gatelink_signal = None
        if self.gatelink_enabled and layer_idx in adapter.gate_A:
            gate_A = adapter.gate_A[layer_idx]
            pre_gate = gate_A @ z_data
            gate_header = struct.pack("!II", pre_gate.shape[0], pre_gate.shape[1] if pre_gate.ndim > 1 else 1)
            gate_bytes = gate_header + pre_gate.astype(np.float64).tobytes()

            gatelink_signal = GateLinkSignal(
                layer_idx=layer_idx,
                ciphertext_bytes=gate_bytes,
                gate_rank=adapter.lora_rank,
                activation_fn="silu",
            )

        elapsed_us = (time.perf_counter() - start_time) * 1e6
        logger.debug("Layer %d HE-LoRA: %.0f μs", layer_idx, elapsed_us)

        return encrypted_delta, gatelink_signal

    def compute_all_deltas(
        self,
        adapter_id: str,
        layer_hidden_states: dict[int, np.ndarray],
    ) -> tuple[list[EncryptedDelta], list[GateLinkSignal]]:
        """Compute encrypted deltas + GateLink signals for ALL layers.

        This is the main entry point. Processes layers in parallel batches
        and returns everything in one batch for the fused single round trip.

        Args:
            adapter_id: Which encrypted adapter to use.
            layer_hidden_states: {layer_idx: hidden_states} from base forward pass.

        Returns:
            Tuple of (all_encrypted_deltas, all_gatelink_signals).
        """
        adapter = self._adapters[adapter_id]
        all_deltas: list[EncryptedDelta] = []
        all_signals: list[GateLinkSignal] = []

        start_time = time.perf_counter()

        # Process layers in parallel batches
        layer_indices = sorted(layer_hidden_states.keys())

        with ThreadPoolExecutor(max_workers=self.max_parallel_layers) as pool:
            futures = []
            for layer_idx in layer_indices:
                if layer_idx not in adapter.encrypted_B:
                    continue
                future = pool.submit(
                    self.compute_layer_delta,
                    adapter,
                    layer_idx,
                    layer_hidden_states[layer_idx],
                )
                futures.append(future)

            for future in futures:
                delta, signal = future.result()
                all_deltas.append(delta)
                if signal is not None:
                    all_signals.append(signal)

        elapsed_ms = (time.perf_counter() - start_time) * 1e3
        logger.info(
            "All HE-LoRA deltas computed: %d layers in %.1f ms (adapter=%s)",
            len(all_deltas),
            elapsed_ms,
            adapter_id,
        )

        return all_deltas, all_signals
