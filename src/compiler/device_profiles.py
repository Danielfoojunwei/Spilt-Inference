"""Device capability profiles for adaptive split compilation.

Defines pre-configured profiles for different device types. Each profile
specifies the optimal split parameters (K, epsilon, rank, etc.) that
balance privacy, performance, and resource constraints.

The compiler uses these profiles to generate device-specific split schedules.
"""

from __future__ import annotations

from src.common.config import DeviceProfile

# Pre-configured device profiles
# Each trades off privacy strength vs. client resource usage vs. throughput
DEVICE_PROFILES: dict[str, DeviceProfile] = {
    # Phone: minimal client compute, strong DP noise, small adapter rank
    # Client runs: embedding + 1 layer + LM head (~1.5 GB)
    # Privacy: epsilon=1.0 (strong noise), rank-4 adapter
    "phone": DeviceProfile(
        name="phone",
        num_client_layers=1,
        epsilon=1.0,
        lora_rank=4,
        max_client_ram_gb=1.5,
        max_rotations_per_token=4,
        has_tee=False,
        speculative_k=1,  # No speculation (save client compute)
    ),
    # Laptop: moderate client compute, moderate DP noise
    # Client runs: embedding + 1 layer + LM head (~3 GB)
    # Privacy: epsilon=4.0, rank-8 adapter
    "laptop": DeviceProfile(
        name="laptop",
        num_client_layers=1,
        epsilon=4.0,
        lora_rank=8,
        max_client_ram_gb=3.0,
        max_rotations_per_token=8,
        has_tee=False,
        speculative_k=2,  # Light speculation
    ),
    # Workstation: more client layers for better privacy mixing
    # Client runs: embedding + 2 layers + LM head (~6 GB)
    # Privacy: epsilon=8.0 (lighter noise, more layers = better mixing)
    "workstation": DeviceProfile(
        name="workstation",
        num_client_layers=2,
        epsilon=8.0,
        lora_rank=16,
        max_client_ram_gb=6.0,
        max_rotations_per_token=16,
        has_tee=False,
        speculative_k=4,  # Full speculation
    ),
    # Server without TEE: max client layers, minimal noise
    # Client runs: embedding + 4 layers + LM head (~16 GB)
    "server": DeviceProfile(
        name="server",
        num_client_layers=4,
        epsilon=16.0,
        lora_rank=32,
        max_client_ram_gb=16.0,
        max_rotations_per_token=64,
        has_tee=False,
        speculative_k=8,
    ),
    # Server with TEE: no DP noise needed (TEE protects input)
    # This matches TenSafe's current architecture
    "server_tee": DeviceProfile(
        name="server_tee",
        num_client_layers=0,  # TEE handles everything
        epsilon=float("inf"),  # No DP noise (TEE provides input privacy)
        lora_rank=32,
        max_client_ram_gb=0.0,
        max_rotations_per_token=64,
        has_tee=True,
        speculative_k=8,
    ),
}


def get_profile(device_type: str) -> DeviceProfile:
    """Get device profile by name.

    Args:
        device_type: One of "phone", "laptop", "workstation", "server", "server_tee".

    Returns:
        DeviceProfile with optimized split parameters.

    Raises:
        ValueError: If device_type is not recognized.
    """
    if device_type not in DEVICE_PROFILES:
        raise ValueError(
            f"Unknown device type '{device_type}'. "
            f"Available: {list(DEVICE_PROFILES.keys())}"
        )
    return DEVICE_PROFILES[device_type]


def auto_detect_profile(
    available_ram_gb: float,
    has_tee: bool = False,
) -> DeviceProfile:
    """Auto-detect the best profile based on available resources.

    Args:
        available_ram_gb: Available RAM on the client device.
        has_tee: Whether the device has TEE capability.

    Returns:
        Best-fit DeviceProfile.
    """
    if has_tee:
        return DEVICE_PROFILES["server_tee"]

    if available_ram_gb >= 12.0:
        return DEVICE_PROFILES["server"]
    elif available_ram_gb >= 4.0:
        return DEVICE_PROFILES["workstation"]
    elif available_ram_gb >= 2.0:
        return DEVICE_PROFILES["laptop"]
    else:
        return DEVICE_PROFILES["phone"]
