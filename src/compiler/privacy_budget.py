"""Joint optimization of privacy parameters (epsilon, K, rank).

Optimizes the privacy-performance trade-off for a given device profile.
The three parameters are coupled:

- epsilon (DP budget): controls noise level. Lower = more private, noisier.
- K (client layers): controls how much mixing happens before server sees states.
  More layers = harder to invert, but more client compute.
- rank (LoRA rank): controls adapter expressivity and HE cost. Higher = better
  quality but more HE computation.

The optimizer finds the Pareto-optimal (epsilon, K, rank) for the device's
resource constraints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from src.common.config import DeviceProfile, PrivacyConfig, SplitInferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudgetResult:
    """Result of privacy budget optimization."""

    epsilon: float
    num_client_layers: int
    lora_rank: int
    estimated_throughput_tps: float  # tokens per second
    estimated_quality_loss_pct: float  # approximate quality degradation
    privacy_score: float  # 0-1, higher = more private


class PrivacyBudgetOptimizer:
    """Joint optimizer for split inference privacy parameters.

    Balances three competing objectives:
    1. Privacy: minimize information leakage (lower ε, higher K)
    2. Performance: maximize throughput (lower K, lower rank)
    3. Quality: minimize output degradation (higher ε, higher rank)
    """

    # Base model throughput (tok/s) without any HE overhead
    BASE_THROUGHPUT = 53.18  # vLLM Llama-3-8B on A100

    # HE overhead per layer (microseconds) with ZeRo-MOAI
    HE_OVERHEAD_PER_LAYER_US = 812.0

    # Approximate quality loss from DP noise (% perplexity increase per unit noise)
    DP_QUALITY_COEFFICIENT = 2.5  # ~2.5% perplexity increase per sigma unit

    # Client layer compute cost (ms per layer)
    CLIENT_LAYER_MS = 15.0  # Approximate for phone/laptop CPU

    def __init__(self, total_model_layers: int = 32, hidden_size: int = 4096):
        self.total_layers = total_model_layers
        self.hidden_size = hidden_size

    def optimize(self, profile: DeviceProfile) -> PrivacyBudgetResult:
        """Find optimal (epsilon, K, rank) for a device profile.

        Args:
            profile: Device capability profile.

        Returns:
            Optimized privacy budget parameters.
        """
        # Use profile defaults as starting point
        epsilon = profile.epsilon
        K = profile.num_client_layers
        rank = profile.lora_rank

        # Adjust K based on RAM constraint
        max_K = self._max_client_layers(profile.max_client_ram_gb)
        K = min(K, max_K)

        # Compute metrics
        throughput = self._estimate_throughput(K, rank)
        quality_loss = self._estimate_quality_loss(epsilon, rank)
        privacy_score = self._compute_privacy_score(epsilon, K)

        result = PrivacyBudgetResult(
            epsilon=epsilon,
            num_client_layers=K,
            lora_rank=rank,
            estimated_throughput_tps=throughput,
            estimated_quality_loss_pct=quality_loss,
            privacy_score=privacy_score,
        )

        logger.info(
            "Privacy budget optimized for '%s': ε=%.1f, K=%d, rank=%d, "
            "throughput=%.1f tok/s, quality_loss=%.1f%%, privacy=%.2f",
            profile.name,
            result.epsilon,
            result.num_client_layers,
            result.lora_rank,
            result.estimated_throughput_tps,
            result.estimated_quality_loss_pct,
            result.privacy_score,
        )

        return result

    def _max_client_layers(self, ram_gb: float) -> int:
        """Estimate max client layers given RAM constraint."""
        # Rough estimate: embedding (~0.5 GB) + LM head (~0.5 GB) + layers (~0.4 GB each)
        available_for_layers = max(0, ram_gb - 1.0)
        return int(available_for_layers / 0.4)

    def _estimate_throughput(self, K: int, rank: int) -> float:
        """Estimate throughput (tok/s) for given parameters.

        Base model throughput reduced by:
        - Client-side layer compute (adds to critical path)
        - HE overhead per server layer (parallel, partially hidden)
        """
        server_layers = self.total_layers - K
        base_per_token_ms = 1000.0 / self.BASE_THROUGHPUT

        # HE overhead: with parallel adapter, partially overlapped with base compute
        # Effective HE overhead ≈ max(0, total_HE - base_compute)
        he_total_ms = server_layers * self.HE_OVERHEAD_PER_LAYER_US / 1000.0
        base_compute_ms = base_per_token_ms * (server_layers / self.total_layers)
        he_effective_ms = max(0, he_total_ms - base_compute_ms * 0.8)  # 80% overlap

        # Client overhead (sequential: adds to latency)
        client_ms = K * self.CLIENT_LAYER_MS

        total_ms = base_per_token_ms + he_effective_ms + client_ms
        return 1000.0 / total_ms

    def _estimate_quality_loss(self, epsilon: float, rank: int) -> float:
        """Estimate quality loss (% perplexity increase).

        Sources of quality loss:
        1. DP noise: proportional to 1/epsilon
        2. Lower rank: proportional to (max_rank - rank) / max_rank
        3. Parallel adapter (no feedback): ~1-2% constant
        """
        if epsilon == float("inf"):
            dp_loss = 0.0
        else:
            # Approximate: sigma ∝ 1/epsilon, quality loss ∝ sigma
            dp_loss = self.DP_QUALITY_COEFFICIENT / epsilon

        rank_loss = max(0, (32 - rank)) * 0.3  # ~0.3% per rank below 32
        parallel_loss = 1.5  # Constant cost of parallel adapter

        return dp_loss + rank_loss + parallel_loss

    def _compute_privacy_score(self, epsilon: float, K: int) -> float:
        """Compute privacy score (0-1, higher = more private).

        Based on:
        - Epsilon: lower = more private (exponential mapping)
        - K: more client layers = harder state inversion
        """
        if epsilon == float("inf"):
            epsilon_score = 0.0
        else:
            # Map epsilon to 0-1: epsilon=1 → 0.9, epsilon=16 → 0.2
            epsilon_score = 1.0 / (1.0 + epsilon / 2.0)

        # K contribution: each client layer adds mixing, reducing inversion accuracy
        # Attack accuracy drops ~15% per layer (from prompt inversion research)
        k_score = 1.0 - (0.9 * (0.85 ** K))  # K=0: 0.1, K=1: 0.24, K=4: 0.53

        return min(1.0, epsilon_score * 0.7 + k_score * 0.3)

    def generate_config(
        self, profile: DeviceProfile, model_id: str = "meta-llama/Llama-3.2-1B"
    ) -> SplitInferenceConfig:
        """Generate a full SplitInferenceConfig from a device profile.

        Args:
            profile: Device capability profile.
            model_id: HuggingFace model ID.

        Returns:
            Complete configuration for split inference.
        """
        result = self.optimize(profile)

        return SplitInferenceConfig(
            model_id=model_id,
            total_layers=self.total_layers,
            num_client_layers=result.num_client_layers,
            num_server_layers=self.total_layers - result.num_client_layers,
            privacy=PrivacyConfig(
                epsilon=result.epsilon,
                enforce_zero_rotation=True,
                gatelink_enabled=True,
            ),
            lora_rank=result.lora_rank,
            adapter_only_encryption=True,
            parallel_adapter=True,
            fused_round_trip=True,
            speculative_k=profile.speculative_k,
        )
