"""Configuration for split inference system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PrivacyConfig:
    """Privacy parameters for the three-layer protection scheme.

    Layer 1 (Input): DP noise with budget epsilon
    Layer 2 (Adapter): CKKS encryption with specified parameters
    Layer 3 (Output): Client-side LM head (always enabled)
    """

    # DP noise parameters (Layer 1: input privacy)
    epsilon: float = 4.0  # Privacy budget (lower = more private, noisier)
    delta: float = 1e-5  # Failure probability
    sensitivity: Optional[float] = None  # L2 sensitivity (None = auto-calibrate)
    noise_mechanism: str = "gaussian"  # "gaussian" or "laplace"

    # CKKS parameters (Layer 2: adapter privacy)
    poly_modulus_degree: int = 16384  # N for CKKS (2^14)
    coeff_mod_bit_sizes: list[int] = field(
        default_factory=lambda: [60, 40, 40, 60]  # L=4 levels, matches TenSafe
    )
    scale_bits: int = 40  # 2^40 scale factor
    enforce_zero_rotation: bool = True  # ZeRo-MOAI constraint

    # GateLink parameters
    gatelink_enabled: bool = True
    gate_activation: str = "silu"  # Non-linear function for gate evaluation


@dataclass
class DeviceProfile:
    """Device capability profile for adaptive split compilation."""

    name: str
    num_client_layers: int  # K: layers to run on client
    epsilon: float  # DP privacy budget
    lora_rank: int  # Adapter rank
    max_client_ram_gb: float  # Maximum client-side RAM usage
    max_rotations_per_token: int  # ZeRo-MOAI rotation budget
    has_tee: bool = False  # Whether device has TEE capability
    speculative_k: int = 1  # Number of draft tokens for speculative batching


@dataclass
class SplitInferenceConfig:
    """Top-level configuration for split inference."""

    # Model
    model_id: str = "meta-llama/Llama-3.2-1B"
    total_layers: int = 16  # Auto-detected from model

    # Split parameters (set by compiler or negotiation)
    num_client_layers: int = 1  # K: client runs layers [0, K)
    num_server_layers: int = 15  # Server runs layers [K, N)

    # Privacy
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    # LoRA adapter
    lora_rank: int = 8
    lora_alpha: float = 16.0
    adapter_only_encryption: bool = True  # AOE mode: encrypt weights, not activations

    # Protocol
    server_address: str = "localhost:50051"
    parallel_adapter: bool = True  # Decoupled base + HE-LoRA paths
    fused_round_trip: bool = True  # Single round trip (GateLink-fused)

    # Speculative batching
    speculative_k: int = 1  # Number of draft tokens (1 = no speculation)
    dp_aware_speculation: bool = True  # Draft on clean data, verify on noisy

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    @property
    def server_layer_range(self) -> tuple[int, int]:
        return (self.num_client_layers, self.num_client_layers + self.num_server_layers)
