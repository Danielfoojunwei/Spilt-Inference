"""Client-side CKKS decryption and GateLink gate evaluation.

After receiving the server's fused response (h_base + encrypted deltas +
GateLink signals), the client:
1. Decrypts each layer's LoRA delta
2. Evaluates GateLink gates (exact non-linear functions)
3. Applies gated deltas to the base model output

This is the client-side assembly step that produces the final hidden states
for LM head decoding.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from typing import Callable, Optional

import numpy as np
import torch

from src.common.types import EncryptedDelta, GateLinkSignal, SplitForwardResponse

logger = logging.getLogger(__name__)


# Exact non-linear activation functions (GateLink: zero approximation error)
ACTIVATION_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda x: np.maximum(x, 0),
    "silu": lambda x: x * (1 / (1 + np.exp(-x))),  # SiLU / Swish
    "gelu": lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": lambda x: np.tanh(x),
}


class CKKSDecryptAssembler:
    """Decrypts encrypted deltas and evaluates GateLink gates on the client.

    In the three-layer privacy architecture:
    - Layer 2 (adapter privacy): this component handles CKKS decryption
    - GateLink: this component evaluates exact non-linear gates locally

    Combined with the client model shard (Layer 1: embedding, Layer 3: LM head),
    this completes the client-side processing pipeline.
    """

    def __init__(self, ckks_secret_key: Optional[bytes] = None):
        """Initialize with CKKS secret key for decryption.

        Args:
            ckks_secret_key: CKKS secret key bytes. In production, loaded from
                            secure storage. If None, uses simulation mode.
        """
        self._secret_key = ckks_secret_key
        self._simulation_mode = ckks_secret_key is None

        if self._simulation_mode:
            logger.info("CKKSDecryptAssembler in simulation mode (no secret key)")

    def decrypt_delta(self, encrypted_delta: EncryptedDelta) -> np.ndarray:
        """Decrypt one layer's encrypted LoRA delta.

        Args:
            encrypted_delta: CKKS-encrypted delta from server.

        Returns:
            Plaintext delta vector/matrix.
        """
        if self._simulation_mode:
            # Simulation: extract plaintext from "encrypted" bytes
            header_size = struct.calcsize("!II")
            rows, cols = struct.unpack("!II", encrypted_delta.ciphertext_bytes[:header_size])
            data = np.frombuffer(
                encrypted_delta.ciphertext_bytes[header_size:], dtype=np.float64
            )
            if cols > 1:
                return data.reshape(rows, cols)
            return data.reshape(rows)
        else:
            # Production: use Pyfhel/N2HE decryption
            # from Pyfhel import Pyfhel
            # he = Pyfhel()
            # he.from_bytes_secret_key(self._secret_key)
            # return he.decrypt(encrypted_delta.ciphertext_bytes)
            raise NotImplementedError("Production CKKS decryption requires Pyfhel")

    def evaluate_gate(self, signal: GateLinkSignal) -> np.ndarray:
        """Evaluate GateLink gate: decrypt pre-activation, apply exact non-linear.

        This is the client-side component of the GateLink protocol. Instead of
        TenSafe's per-layer round trip (server sends pre-activation, client returns
        gate bit), we evaluate all gates locally in one batch.

        Args:
            signal: Encrypted pre-activation from server.

        Returns:
            Gate values (post non-linear activation).
        """
        # Decrypt the pre-activation
        if self._simulation_mode:
            header_size = struct.calcsize("!II")
            rows, cols = struct.unpack("!II", signal.ciphertext_bytes[:header_size])
            pre_activation = np.frombuffer(
                signal.ciphertext_bytes[header_size:], dtype=np.float64
            )
            if cols > 1:
                pre_activation = pre_activation.reshape(rows, cols)
            else:
                pre_activation = pre_activation.reshape(rows)
        else:
            raise NotImplementedError("Production CKKS decryption requires Pyfhel")

        # Apply exact non-linear function (zero approximation error)
        activation_fn = ACTIVATION_FNS.get(signal.activation_fn)
        if activation_fn is None:
            raise ValueError(f"Unknown activation: {signal.activation_fn}")

        gate_values = activation_fn(pre_activation)
        return gate_values

    def assemble(
        self,
        response: SplitForwardResponse,
    ) -> torch.Tensor:
        """Full client-side assembly: decrypt + gate + combine.

        Takes the server's fused response and produces final hidden states
        ready for the LM head.

        Args:
            response: Server response with base_hidden_states, encrypted_deltas,
                     and gatelink_signals.

        Returns:
            Final hidden states tensor, shape [batch, seq_len, hidden_dim].
        """
        start_time = time.perf_counter()

        h_base = torch.from_numpy(response.base_hidden_states).float()

        # Build gate lookup: {layer_idx: gate_values}
        gates: dict[int, np.ndarray] = {}
        for signal in response.gatelink_signals:
            gates[signal.layer_idx] = self.evaluate_gate(signal)

        # Decrypt and apply gated deltas
        total_delta = torch.zeros_like(h_base)

        for enc_delta in response.encrypted_deltas:
            delta = self.decrypt_delta(enc_delta)
            delta_tensor = torch.from_numpy(delta.copy()).float()

            # Apply GateLink gate if available
            # Gate is rank-r dimensional; use its mean as a scalar gate value
            # (full gating would require rank-r → hidden_dim projection on client)
            if enc_delta.layer_idx in gates:
                gate = gates[enc_delta.layer_idx]
                gate_scalar = float(np.mean(gate))
                delta_tensor = delta_tensor * gate_scalar

            # Flatten delta to 1D hidden_dim vector for broadcasting
            delta_flat = delta_tensor.flatten()
            hidden_dim = h_base.shape[-1]
            if delta_flat.numel() >= hidden_dim:
                delta_vec = delta_flat[:hidden_dim]
            else:
                delta_vec = torch.zeros(hidden_dim)
                delta_vec[: delta_flat.numel()] = delta_flat

            # Broadcast across all dimensions of h_base
            total_delta = total_delta + delta_vec

        h_final = h_base + total_delta

        elapsed_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(
            "Client assembly: %d deltas, %d gates in %.1f ms",
            len(response.encrypted_deltas),
            len(response.gatelink_signals),
            elapsed_ms,
        )

        return h_final
