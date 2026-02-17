"""Calibrated differential privacy noise for hidden state protection.

Implements the Gaussian mechanism for ε-differential privacy on hidden states
before they are transmitted to the server. This is Layer 1 of the three-layer
privacy architecture (DP for input, HE for adapter, local compute for output).

Based on:
- Split-and-Denoise (SnD, ICML 2024): calibrated noise reduces correlation to <0.005
- DEL (2025): projection + DP quantization for high-dimensional states
- Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# Recommended epsilon values per device tier
# Lower epsilon = more privacy = more noise = more quality loss
RECOMMENDED_EPSILON = {
    "phone": 1.0,  # Strong privacy (most noise)
    "laptop": 4.0,  # Moderate privacy
    "workstation": 8.0,  # Light privacy
    "server": 16.0,  # Minimal noise (TEE recommended instead)
}


@dataclass
class NoiseStats:
    """Statistics about the injected noise for monitoring."""

    sigma: float
    noise_norm: float
    signal_norm: float
    snr_db: float  # Signal-to-noise ratio in dB


class DPNoiseInjector:
    """Calibrated Gaussian noise mechanism for hidden state privacy.

    The Gaussian mechanism adds noise ~ N(0, sigma^2 * I) where:
        sigma = (sensitivity * sqrt(2 * ln(1.25 / delta))) / epsilon

    The L2 sensitivity is the maximum L2 norm change in the output when
    any single input changes. For hidden states after K transformer layers,
    this is bounded by the maximum hidden state norm (auto-calibrated from
    a running estimate if not provided).
    """

    def __init__(
        self,
        epsilon: float = 4.0,
        delta: float = 1e-5,
        sensitivity: Optional[float] = None,
        clip_norm: Optional[float] = None,
    ):
        """Initialize DP noise injector.

        Args:
            epsilon: Privacy budget. Lower = more private, more noise.
                     Recommended: 1.0 (phone), 4.0 (laptop), 8.0 (workstation).
            delta: Failure probability. Typically 1e-5 or 1/n for n data points.
            sensitivity: L2 sensitivity. If None, auto-calibrated from hidden
                        state norms (uses clip_norm as upper bound).
            clip_norm: Maximum L2 norm for hidden states. States exceeding this
                      are clipped before noise addition. If None, estimated from data.
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be in (0, 1), got {delta}")

        self.epsilon = epsilon
        self.delta = delta
        self._fixed_sensitivity = sensitivity
        self._clip_norm = clip_norm

        # Running norm estimate for auto-calibration
        self._norm_sum = 0.0
        self._norm_count = 0
        self._max_observed_norm = 0.0

    @property
    def sensitivity(self) -> float:
        """Get L2 sensitivity (fixed or auto-calibrated)."""
        if self._fixed_sensitivity is not None:
            return self._fixed_sensitivity
        if self._clip_norm is not None:
            return self._clip_norm
        if self._max_observed_norm > 0:
            # Use 1.5x max observed norm as conservative bound
            return self._max_observed_norm * 1.5
        # Default: assume unit norm (will be corrected after first call)
        return 1.0

    @property
    def sigma(self) -> float:
        """Compute noise standard deviation from (epsilon, delta, sensitivity)."""
        return self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def inject_noise(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, NoiseStats]:
        """Add calibrated DP noise to hidden states.

        Args:
            hidden_states: Shape [batch_size, seq_len, hidden_dim].

        Returns:
            Tuple of (noised_states, noise_stats).
            noised_states has the same shape, with calibrated Gaussian noise added.
        """
        # Update norm statistics for auto-calibration
        with torch.no_grad():
            # Compute per-token L2 norms
            norms = torch.norm(
                hidden_states.float(), p=2, dim=-1
            )  # [batch, seq_len]
            max_norm = norms.max().item()
            mean_norm = norms.mean().item()

            self._max_observed_norm = max(self._max_observed_norm, max_norm)
            self._norm_sum += mean_norm
            self._norm_count += 1

        # Clip hidden states if clip_norm is set (bounds sensitivity)
        if self._clip_norm is not None:
            norms_for_clip = norms.unsqueeze(-1).clamp(min=1e-8)
            clip_factor = self._clip_norm / norms_for_clip
            clip_factor = torch.clamp(clip_factor, max=1.0)
            hidden_states = hidden_states * clip_factor

        # Generate calibrated Gaussian noise
        current_sigma = self.sigma
        noise = torch.randn_like(hidden_states) * current_sigma

        # Add noise
        noised_states = hidden_states + noise

        # Compute stats
        noise_norm = torch.norm(noise.float()).item()
        signal_norm = torch.norm(hidden_states.float()).item()
        ratio = signal_norm / (noise_norm + 1e-10)
        snr_db = 20 * math.log10(max(ratio, 1e-10))

        stats = NoiseStats(
            sigma=current_sigma,
            noise_norm=noise_norm,
            signal_norm=signal_norm,
            snr_db=snr_db,
        )

        logger.debug(
            "DP noise: epsilon=%.2f, sigma=%.4f, SNR=%.1f dB",
            self.epsilon,
            current_sigma,
            snr_db,
        )

        return noised_states, stats

    @staticmethod
    def recommended_epsilon(device_type: str) -> float:
        """Get recommended epsilon for a device type.

        Args:
            device_type: One of "phone", "laptop", "workstation", "server".

        Returns:
            Recommended epsilon value.
        """
        return RECOMMENDED_EPSILON.get(device_type, 4.0)

    @staticmethod
    def privacy_guarantee_summary(epsilon: float, delta: float) -> str:
        """Human-readable summary of the privacy guarantee."""
        return (
            f"({epsilon}, {delta})-differential privacy: "
            f"For any two inputs differing in one token, the probability of "
            f"any output from the mechanism changes by at most e^{epsilon} "
            f"= {math.exp(epsilon):.2f}x, except with probability {delta}."
        )
