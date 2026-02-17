"""Benchmark: DP noise impact on signal quality.

Measures signal-to-noise ratio (SNR) and correlation between clean and
noised hidden states at different privacy budgets (epsilon values).

This validates the privacy-utility trade-off:
- Lower epsilon → more noise → worse SNR → better privacy
- Higher epsilon → less noise → better SNR → weaker privacy
"""

from __future__ import annotations

import numpy as np
import torch

from src.client.dp_noise import DPNoiseInjector


def measure_snr_at_epsilon(
    epsilon: float, hidden_dim: int = 4096, seq_len: int = 32, n_trials: int = 50
) -> dict:
    """Measure SNR and correlation at a given epsilon."""
    dp = DPNoiseInjector(epsilon=epsilon, delta=1e-5, sensitivity=10.0)

    snrs = []
    correlations = []

    for _ in range(n_trials):
        h = torch.randn(1, seq_len, hidden_dim) * 5.0  # Realistic norm

        noised, stats = dp.inject_noise(h)
        snrs.append(stats.snr_db)

        # Cosine similarity between clean and noised
        h_flat = h.flatten().numpy()
        n_flat = noised.flatten().numpy()
        cos_sim = np.dot(h_flat, n_flat) / (np.linalg.norm(h_flat) * np.linalg.norm(n_flat) + 1e-10)
        correlations.append(cos_sim)

    return {
        "epsilon": epsilon,
        "mean_snr_db": np.mean(snrs),
        "std_snr_db": np.std(snrs),
        "mean_correlation": np.mean(correlations),
        "std_correlation": np.std(correlations),
        "sigma": dp.sigma,
    }


def main():
    print("=" * 70)
    print("DP Noise Utility Benchmark")
    print("=" * 70)
    print()
    print(f"{'Epsilon':>8} {'Sigma':>10} {'SNR (dB)':>12} {'Correlation':>14} {'Privacy':>10}")
    print("-" * 70)

    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    for eps in epsilons:
        result = measure_snr_at_epsilon(eps)
        privacy = "Strong" if eps <= 1 else "Moderate" if eps <= 4 else "Light" if eps <= 16 else "Weak"
        print(
            f"{result['epsilon']:>8.1f} "
            f"{result['sigma']:>10.4f} "
            f"{result['mean_snr_db']:>8.1f} ± {result['std_snr_db']:>4.1f} "
            f"{result['mean_correlation']:>8.4f} ± {result['std_correlation']:>4.4f} "
            f"{privacy:>10}"
        )

    print()
    print("Interpretation:")
    print("  - SNR > 20 dB: DP noise barely perceptible, good utility")
    print("  - SNR 10-20 dB: Noticeable noise, moderate utility loss")
    print("  - SNR < 10 dB: Heavy noise, significant utility loss")
    print("  - Correlation < 0.005: Strong privacy (SnD ICML'24 target)")
    print("  - Correlation > 0.5: Weak privacy, hidden states still informative")


if __name__ == "__main__":
    main()
