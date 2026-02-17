"""Tests for DP noise injection module."""

import math

import numpy as np
import pytest
import torch

from src.client.dp_noise import RECOMMENDED_EPSILON, DPNoiseInjector


class TestDPNoiseInjector:
    """Test calibrated Gaussian mechanism for hidden state privacy."""

    def test_basic_noise_injection(self):
        """Noise is added and shapes are preserved."""
        dp = DPNoiseInjector(epsilon=4.0, delta=1e-5)
        h = torch.randn(1, 10, 128)  # [batch, seq_len, hidden_dim]

        noised, stats = dp.inject_noise(h)

        assert noised.shape == h.shape
        assert not torch.allclose(noised, h), "Noise should change the tensor"
        assert stats.sigma > 0
        assert stats.noise_norm > 0

    def test_epsilon_affects_noise_level(self):
        """Lower epsilon → more noise (stronger privacy)."""
        h = torch.randn(1, 10, 128)

        dp_strong = DPNoiseInjector(epsilon=1.0, delta=1e-5, sensitivity=10.0)
        dp_weak = DPNoiseInjector(epsilon=16.0, delta=1e-5, sensitivity=10.0)

        assert dp_strong.sigma > dp_weak.sigma, "Lower epsilon should give higher sigma"

        _, stats_strong = dp_strong.inject_noise(h)
        _, stats_weak = dp_weak.inject_noise(h)

        assert stats_strong.snr_db < stats_weak.snr_db, "Stronger privacy = lower SNR"

    def test_sigma_formula(self):
        """Verify Gaussian mechanism sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon."""
        epsilon = 4.0
        delta = 1e-5
        sensitivity = 10.0

        dp = DPNoiseInjector(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

        expected_sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        assert abs(dp.sigma - expected_sigma) < 1e-10

    def test_noise_is_gaussian(self):
        """Verify noise distribution is approximately Gaussian."""
        dp = DPNoiseInjector(epsilon=4.0, delta=1e-5, sensitivity=1.0)
        h = torch.zeros(1, 1000, 128)  # Large tensor for statistical test

        noised, _ = dp.inject_noise(h)
        noise = (noised - h).flatten().numpy()

        # Check mean ≈ 0
        assert abs(noise.mean()) < 0.1, f"Noise mean should be ~0, got {noise.mean():.4f}"

        # Check std ≈ sigma
        expected_std = dp.sigma
        actual_std = noise.std()
        assert abs(actual_std - expected_std) / expected_std < 0.1, (
            f"Noise std should be ~{expected_std:.4f}, got {actual_std:.4f}"
        )

    def test_clip_norm(self):
        """Clipping bounds sensitivity."""
        clip = 5.0
        dp = DPNoiseInjector(epsilon=4.0, delta=1e-5, clip_norm=clip)

        # Create tensor with large norms
        h = torch.randn(1, 10, 128) * 100

        noised, stats = dp.inject_noise(h)

        # Sensitivity should equal clip_norm
        assert dp.sensitivity == clip

    def test_auto_calibration(self):
        """Auto-calibration updates sensitivity from observed norms."""
        dp = DPNoiseInjector(epsilon=4.0, delta=1e-5)

        # First call: sensitivity defaults
        h1 = torch.randn(1, 10, 128) * 5.0
        dp.inject_noise(h1)

        # After observing data, max_observed_norm should be updated
        assert dp._max_observed_norm > 0

    def test_invalid_epsilon(self):
        """Epsilon must be positive."""
        with pytest.raises(ValueError, match="positive"):
            DPNoiseInjector(epsilon=0.0)

        with pytest.raises(ValueError, match="positive"):
            DPNoiseInjector(epsilon=-1.0)

    def test_invalid_delta(self):
        """Delta must be in (0, 1)."""
        with pytest.raises(ValueError):
            DPNoiseInjector(epsilon=1.0, delta=0.0)

        with pytest.raises(ValueError):
            DPNoiseInjector(epsilon=1.0, delta=1.0)

    def test_recommended_epsilon(self):
        """Recommended epsilon values for each device tier."""
        assert DPNoiseInjector.recommended_epsilon("phone") == 1.0
        assert DPNoiseInjector.recommended_epsilon("laptop") == 4.0
        assert DPNoiseInjector.recommended_epsilon("workstation") == 8.0
        assert DPNoiseInjector.recommended_epsilon("server") == 16.0
        assert DPNoiseInjector.recommended_epsilon("unknown") == 4.0  # Default

    def test_privacy_guarantee_summary(self):
        """Human-readable privacy summary."""
        summary = DPNoiseInjector.privacy_guarantee_summary(4.0, 1e-5)
        assert "differential privacy" in summary
        assert "4.0" in summary

    def test_deterministic_with_seed(self):
        """Same seed produces same noise."""
        dp = DPNoiseInjector(epsilon=4.0, delta=1e-5, sensitivity=1.0)
        h = torch.randn(1, 10, 128)

        torch.manual_seed(42)
        noised1, _ = dp.inject_noise(h.clone())

        torch.manual_seed(42)
        noised2, _ = dp.inject_noise(h.clone())

        assert torch.allclose(noised1, noised2)
