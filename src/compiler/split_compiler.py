"""Split inference compiler: generates device-specific split schedules.

Extends TenSafe's cost-budgeted compiler with:
- Privacy budget parameters (epsilon, K, device profile)
- Adapter-Only Encryption (AOE) scheduling
- GateLink-fused round trip optimization
- DP-aware speculative batching configuration

The compiler takes a model + device profile and produces a complete
SplitInferenceConfig that optimizes for the device's constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.common.config import DeviceProfile, SplitInferenceConfig
from src.compiler.device_profiles import DEVICE_PROFILES, auto_detect_profile, get_profile
from src.compiler.privacy_budget import PrivacyBudgetOptimizer, PrivacyBudgetResult

logger = logging.getLogger(__name__)


@dataclass
class CompilationReport:
    """Report from the split compiler with optimization details."""

    config: SplitInferenceConfig
    budget: PrivacyBudgetResult
    profile: DeviceProfile
    warnings: list[str]


class SplitCompiler:
    """Compiler for split inference schedules.

    Takes a model specification and device profile, then generates an
    optimized SplitInferenceConfig including:
    - Split point (K client layers)
    - Privacy parameters (epsilon, delta)
    - LoRA configuration (rank, alpha, AOE mode)
    - Protocol settings (fused round trip, parallel adapter)
    - Speculative batching parameters
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B",
        total_layers: int = 16,
        hidden_size: int = 2048,
    ):
        self.model_id = model_id
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self._optimizer = PrivacyBudgetOptimizer(
            total_model_layers=total_layers,
            hidden_size=hidden_size,
        )

    def compile(
        self,
        device_type: str = "laptop",
        server_address: str = "localhost:50051",
    ) -> CompilationReport:
        """Compile a split inference schedule for a device type.

        Args:
            device_type: One of "phone", "laptop", "workstation", "server", "server_tee".
            server_address: gRPC server address.

        Returns:
            CompilationReport with optimized config and diagnostics.
        """
        profile = get_profile(device_type)
        return self._compile_for_profile(profile, server_address)

    def compile_auto(
        self,
        available_ram_gb: float,
        has_tee: bool = False,
        server_address: str = "localhost:50051",
    ) -> CompilationReport:
        """Auto-detect device profile and compile.

        Args:
            available_ram_gb: Available client RAM in GB.
            has_tee: Whether device has TEE.
            server_address: gRPC server address.

        Returns:
            CompilationReport with auto-detected profile.
        """
        profile = auto_detect_profile(available_ram_gb, has_tee)
        logger.info("Auto-detected profile: %s", profile.name)
        return self._compile_for_profile(profile, server_address)

    def _compile_for_profile(
        self, profile: DeviceProfile, server_address: str
    ) -> CompilationReport:
        """Core compilation logic for a given profile."""
        warnings: list[str] = []

        # Optimize privacy budget
        budget = self._optimizer.optimize(profile)

        # Generate config
        config = self._optimizer.generate_config(profile, self.model_id)
        config.server_address = server_address

        # Validate constraints
        if budget.estimated_quality_loss_pct > 10.0:
            warnings.append(
                f"High estimated quality loss ({budget.estimated_quality_loss_pct:.1f}%). "
                f"Consider increasing epsilon or LoRA rank."
            )

        if budget.estimated_throughput_tps < 5.0:
            warnings.append(
                f"Low estimated throughput ({budget.estimated_throughput_tps:.1f} tok/s). "
                f"Consider reducing client layers or LoRA rank."
            )

        if profile.num_client_layers == 0 and not profile.has_tee:
            warnings.append(
                "No client layers and no TEE. Server will see raw embeddings. "
                "Input privacy relies solely on DP noise."
            )

        estimated_client_ram = ClientModelShard_estimate_memory(
            self.hidden_size, profile.num_client_layers, 32000  # Approximate vocab size
        )
        if estimated_client_ram > profile.max_client_ram_gb:
            warnings.append(
                f"Estimated client RAM ({estimated_client_ram:.1f} GB) exceeds "
                f"device limit ({profile.max_client_ram_gb:.1f} GB). "
                f"Reducing client layers."
            )
            config.num_client_layers = max(0, config.num_client_layers - 1)
            config.num_server_layers = self.total_layers - config.num_client_layers

        report = CompilationReport(
            config=config,
            budget=budget,
            profile=profile,
            warnings=warnings,
        )

        logger.info(
            "Compilation complete for '%s': K=%d, ε=%.1f, rank=%d, "
            "throughput=%.1f tok/s, privacy=%.2f, warnings=%d",
            profile.name,
            config.num_client_layers,
            config.privacy.epsilon,
            config.lora_rank,
            budget.estimated_throughput_tps,
            budget.privacy_score,
            len(warnings),
        )

        for w in warnings:
            logger.warning("Compiler warning: %s", w)

        return report


def ClientModelShard_estimate_memory(
    hidden_size: int, num_layers: int, vocab_size: int
) -> float:
    """Estimate client shard memory (GB). Mirrors ClientModelShard.estimate_memory_gb."""
    bytes_per_param = 4
    embed_params = vocab_size * hidden_size
    layer_params = num_layers * (4 * hidden_size * hidden_size)
    norm_params = hidden_size
    lm_head_params = hidden_size * vocab_size
    total_params = embed_params + layer_params + norm_params + lm_head_params
    return (total_params * bytes_per_param) / (1024**3)
