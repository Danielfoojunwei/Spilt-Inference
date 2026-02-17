"""Shared type definitions for the split inference protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EncryptedDelta:
    """A CKKS-encrypted LoRA delta for one transformer layer.

    In AOE mode: encrypted_B @ plaintext_x → encrypted delta.
    The client decrypts this to get the plaintext LoRA correction.
    """

    layer_idx: int
    ciphertext_bytes: bytes
    num_elements: int  # Number of real-valued elements packed in the ciphertext
    scale: float = 1.0  # CKKS scale factor (2^40 typical)


@dataclass
class GateLinkSignal:
    """Encrypted pre-activation for GateLink gate evaluation.

    The server computes A_gate @ z (rank-r vector) under CKKS and sends it
    to the client. The client decrypts and evaluates the exact non-linear
    function (ReLU/SiLU/GELU) to produce the gate value.

    This replaces TenSafe's per-layer round trip with a batched single response.
    """

    layer_idx: int
    ciphertext_bytes: bytes
    gate_rank: int  # Dimension of the pre-activation vector (same as LoRA rank r)
    activation_fn: str = "silu"  # Non-linear function to evaluate client-side


@dataclass
class SplitForwardRequest:
    """Client → Server request for one forward pass step.

    Contains DP-noised hidden states from the client's K local layers.
    """

    hidden_states: np.ndarray  # Shape: [seq_len, hidden_dim], DP-noised
    token_positions: list[int]  # Position IDs for the tokens
    adapter_id: str = "default"  # Which encrypted adapter to use
    sequence_id: str = ""  # For KV-cache tracking across tokens


@dataclass
class SplitForwardResponse:
    """Server → Client response containing all data for one step.

    Batches base model output, all encrypted LoRA deltas, and all GateLink
    pre-activations into a single response (fused single round trip).
    """

    base_hidden_states: np.ndarray  # Shape: [seq_len, hidden_dim], base model output
    encrypted_deltas: list[EncryptedDelta] = field(default_factory=list)
    gatelink_signals: list[GateLinkSignal] = field(default_factory=list)
    layers_computed: int = 0  # Number of server-side layers executed


@dataclass
class NegotiateRequest:
    """Client announces device capabilities for split parameter negotiation."""

    device_type: str  # "phone", "laptop", "workstation", "server"
    available_ram_gb: float
    has_tee: bool = False
    preferred_epsilon: Optional[float] = None
    max_client_layers: Optional[int] = None


@dataclass
class NegotiateResponse:
    """Server responds with optimized split parameters."""

    num_client_layers: int  # K: how many layers client should run
    epsilon: float  # DP noise budget
    lora_rank: int  # Adapter rank for this session
    adapter_id: str  # Which pre-encrypted adapter to use
    total_model_layers: int  # N: total layers in the model


@dataclass
class UploadAdapterRequest:
    """Client uploads encrypted adapter weights (one-time, AOE mode).

    Only B matrices are encrypted (FFA-LoRA: A is frozen/public).
    Encrypted once, stored on server, used for all subsequent requests.
    """

    adapter_id: str
    encrypted_B_matrices: dict[int, bytes]  # {layer_idx: CKKS ciphertext of B matrix}
    plaintext_A_matrices: dict[int, np.ndarray]  # {layer_idx: plaintext A matrix}
    gate_A_matrices: dict[int, np.ndarray]  # {layer_idx: GateLink gate projection}
    lora_rank: int
    lora_alpha: float
    num_layers: int
