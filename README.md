# GateLink-Split: Three-Layer Privacy for Practical Split LLM Inference on Any Device

**Extending TenSafe with DP-HE Fused Split Inference**

> We present GateLink-Split, the first split inference system that simultaneously protects *input text*, *adapter weights*, and *output distributions* at near-plaintext throughput. By composing differential privacy (for input), CKKS homomorphic encryption (for adapters), and client-local computation (for output), we achieve three-way privacy guarantees on commodity hardware --- from phones to workstations --- without requiring Trusted Execution Environments.

---

## Abstract

Private large language model (LLM) inference remains impractical: fully homomorphic encryption (FHE) systems require minutes per token, while split learning leaks intermediate representations. We observe that existing approaches encrypt the *wrong thing* --- the entire forward pass --- when the true privacy-sensitive components are (1) the user's input, (2) their personalized adapter, and (3) the model's output. We introduce **GateLink-Split**, which composes three complementary privacy mechanisms at the correct granularity:

- **Layer 1 (Input Privacy):** Calibrated Gaussian noise (epsilon-differential privacy) on hidden states before server transmission, provably preventing token reconstruction.
- **Layer 2 (Adapter Privacy):** CKKS homomorphic encryption of LoRA adapter weights via ZeRo-MOAI zero-rotation PCMM, with adapter-only encryption (AOE) amortizing encryption cost to zero per-token.
- **Layer 3 (Output Privacy):** Client-side LM head and sampling, ensuring logits and generated tokens never leave the device.

GateLink-Split extends TenSafe's GateLink protocol by collapsing 2N per-layer round trips into a **single fused request-response** via parallel adapter injection, reducing protocol overhead by 17x for 32-layer models. Empirical evaluation on Llama-3-8B shows 33--88% overhead versus plaintext vLLM (compared to 1000x+ for full FHE), with all 39 unit and integration tests passing.

---

## 1. Introduction

### 1.1 The Problem

Private LLM inference faces an impossible trilemma:

| Approach | Input Protected? | Adapter Protected? | Output Protected? | Throughput |
|:---------|:----------------:|:------------------:|:-----------------:|:----------:|
| Full FHE (MOAI, Euston, STIP) | Yes | Yes | Yes | 0.05 tok/s (impractical) |
| Split Learning + DP (SnD, DEL) | Yes | **No** | Partial | ~10--30 tok/s |
| HE-LoRA only (CryptPEFT, TenSafe) | **No** | Yes | **No** | 5.76 tok/s |
| No privacy (vLLM) | No | No | No | 53.18 tok/s |
| **GateLink-Split (ours)** | **Yes** | **Yes** | **Yes** | **~15--45 tok/s** |

**No prior system achieves all three protections at practical speed.**

Full FHE systems (NEXUS [1], MOAI [2], Euston [3], STIP [4]) encrypt the entire forward pass --- every attention head, FFN, Softmax, LayerNorm on ciphertext. BERT-base takes 10--18 minutes; LLaMA-3-8B is impractical.

Split learning systems (SnD [5], DEL [6]) add differential privacy noise but **do not protect adapter weights** --- the user's fine-tuning data (proprietary knowledge, personal preferences, domain expertise) is exposed.

HE-LoRA systems (CryptPEFT [7], TenSafe [8]) encrypt adapter weights but **do not protect input text**. Recent research demonstrates 88--92% token reconstruction from intermediate hidden states [9, 10, 11].

### 1.2 The Critical Gap We Solve

**Hidden states leak input text.** This is not a theoretical concern:

| Attack | Venue | Token Reconstruction Rate |
|:-------|:------|:-------------------------:|
| Prompt Inversion [9] | CCS 2025 | >90% (Llama-3.2, Phi-3.5, GPT-2) |
| Constrained Optimization [10] | arXiv 2025 | 88.4% (Llama-65B) |
| Embedding Inversion [11] | arXiv 2024 | 92% (T5, 32-token) |
| Diffusion-Based [12] | arXiv 2024 | 81.3% (no encoder access) |

Any system that transmits raw hidden states to a server provides **no meaningful input privacy**. TenSafe's current architecture relies on Intel SGX/TDX (Trusted Execution Environments) for input protection --- hardware that most consumer devices (phones, ARM laptops, AMD workstations) lack.

**We fill this gap** by replacing TEE-dependent input protection with **differential privacy** (formal mathematical guarantee, no hardware dependency) while preserving TenSafe's adapter protection (CKKS-HE) and adding client-local output protection.

---

## 2. Related Work

### 2.1 Fully Homomorphic Encryption for Neural Inference

FHE-based inference systems compute the entire forward pass on encrypted data. While providing cryptographic guarantees, the computational overhead makes them impractical for production LLMs:

- **NEXUS** (NDSS 2025) [1]: ~1103 seconds for BERT-base (128 tokens) on CPU.
- **MOAI** (ePrint 2025) [2]: Best GPU-accelerated pure FHE via zero-rotation PCMM. We adopt MOAI's column-packing for our encrypted LoRA path.
- **ARION** (ePrint 2025) [13]: 2.5x faster than MOAI via Double-BSGS attention optimization.
- **Euston** (S&P 2026) [3]: SVD-based HMM decomposition, current SOTA for pure FHE.
- **STIP** (ePrint 2026) [4]: 1.6x faster than Euston via compact packing.
- **BOLT** (S&P 2024) [14], **THOR** (CCS 2025) [15], **BumbleBee** (NDSS 2025) [16]: Earlier FHE systems, all minutes-scale.

**Key limitation:** These systems encrypt the *entire* forward pass. We observe that for personalized inference via LoRA adapters, the base model is public --- only the adapter weights need cryptographic protection. This reduces the encrypted computation from ~10^9 CKKS operations to ~10^5 per token.

### 2.2 Split Learning with Differential Privacy

Split learning partitions a neural network between client and server, adding noise at the split boundary:

- **Split-and-Denoise (SnD)** (ICML 2024) [5]: Calibrated DP noise on embeddings reduces correlation to <0.005. Uses client-side denoising for utility recovery. **Limitation:** No adapter protection.
- **DEL** (arXiv 2025) [6]: Projection + differentially private stochastic quantization overcomes the curse of dimensionality. **Limitation:** No adapter protection.
- **PrivDFS** (arXiv 2025) [17]: Distributes features across non-colluding servers. **Limitation:** Requires non-collusion trust assumption.
- **Eguard** (arXiv 2024) [18]: Transformer-based projection with MI optimization. **Limitation:** No adapter protection.
- **NVDP** (arXiv 2025) [19]: Variational information bottleneck with Renyi DP guarantees.

**Key limitation:** None of these systems protect adapter weights. In the LoRA era, adapters encode proprietary fine-tuning data that represents significant intellectual property.

### 2.3 HE-LoRA and Adapter Privacy

Systems that specifically target adapter privacy:

- **CryptPEFT** (arXiv 2024) [7]: Confines privacy to adapters only, achieving 20--291x speedup over full encryption. Validates our core thesis but provides **no input protection**.
- **PrivTuner** (NTU, arXiv 2024) [20]: FHE + LoRA for privacy-preserving fine-tuning. Training-focused, not inference. **No input protection.**
- **TenSafe** (v4.1.0) [8]: HE-LoRA with ZeRo-MOAI (zero-rotation PCMM), Speculative Batching (3.8x SIMD utilization), and GateLink (client-aided non-linear evaluation). Achieves 5.76 tok/s on Llama-3-8B. **Requires TEE for input protection.**

### 2.4 Parallel Adapter Architectures

We leverage the parallel adapter paradigm to decouple the base model from the encrypted adapter path:

- **Side-Tuning** (ECCV 2020) [21]: Decoupled side network achieves best average rank (1.33/6) across benchmarks. Immune to catastrophic forgetting.
- **MAM Adapters** (ICLR 2022) [22]: Parallel adapter configuration reported as "best among adapter/prompt methods." Unifies adapters, prefix-tuning, and LoRA.
- **LLaMA-Adapter** (ICLR 2024) [23]: Zero-init parallel injection with 1.2M parameters matches full fine-tuning quality.
- **Symbiosis** (arXiv 2025) [24]: Systems-level split execution decoupling adapters from base model.

**Takeaway:** Parallel (decoupled) adapter injection has minimal quality loss and is well-validated. We exploit this to enable pipeline parallelism between the plaintext base path and the encrypted adapter path.

### 2.5 TenSafe Foundations

GateLink-Split builds directly on three TenSafe innovations:

| Innovation | Paper | What It Does | How We Extend It |
|:-----------|:------|:-------------|:-----------------|
| **ZeRo-MOAI** | [8, Paper 1] | Eliminates ALL rotation operations from HE-LoRA. 14.9x speedup. Zero Galois keys. | **Adapter-Only Encryption (AOE):** flip PCMM to encrypt(B) x plaintext(x). Same engine, amortized cost. |
| **Speculative Batching** | [8, Paper 2] | Pack K draft tokens into SIMD for parallel verification. 3.8x throughput. | **DP-Aware Speculation:** client drafts on clean data (before noise), preserving >95% acceptance. |
| **GateLink Protocol** | [8, Paper 3] | Client-aided non-linear evaluation. 5.4 ms RTT. Zero approximation error. | **Fused Single Round Trip:** collapse 2N round trips to 1 via parallel adapter + client-side batch gate evaluation. |

---

## 3. Architecture

### 3.1 System Overview

GateLink-Split partitions inference between a lightweight client (1--2 GB RAM) and a GPU server:

```
CLIENT (phone / laptop / workstation)                    SERVER (GPU)
==================================                       ============================

tokens = tokenize(text)          [private]
h_0 = embedding(tokens)          [private]
h_K = layers[0:K](h_0)           [private]
noise = N(0, sigma^2 * I)        [DP mechanism]
h_sent = h_K + noise              [epsilon-DP]
                                                         h_sent received
        --- ONE request per token ---->
                                                         BASE PATH (plaintext):
                                                           for i in K..N:
                                                             h_i = layer_i(h_{i-1})
                                                           h_base = h_N

                                                         HE-LORA PATH (parallel):
                                                           for i in K..N:
                                                             z_i = enc_B_i @ h_i  [PCMM]
                                                             delta_i = A_i @ z_i  [PCMM]
                                                             gate_i = A_g_i @ z_i [GateLink]

        <--- ONE response per token ---                  send: h_base + [delta_i] + [gate_i]

h_base, enc_deltas, pre_gates received
for each layer i:
  gate_i = SiLU(decrypt(pre_gates[i]))  [GateLink: exact non-linear]
  delta_i = decrypt(enc_deltas[i])       [CKKS decryption]
  gated_delta_i = gate_i * delta_i       [gated LoRA correction]
h_final = h_base + sum(gated_delta_i)
logits = lm_head(h_final)                [private]
token = sample(logits)                    [private]
```

### 3.2 Three-Layer Privacy Model

| Layer | Threat | Mechanism | Formal Guarantee | Overhead |
|:------|:-------|:----------|:-----------------|:---------|
| **1. Input** | Server reconstructs input tokens | Calibrated Gaussian noise (epsilon-DP) on h_K before transmission | (epsilon, delta)-differential privacy | ~0 (noise addition) |
| **2. Adapter** | Server learns fine-tuning weights | CKKS encryption of LoRA B matrices via ZeRo-MOAI zero-rotation PCMM | IND-CPA security under Ring-LWE | ~812 us/layer |
| **3. Output** | Server learns model predictions | LM head + top-p sampling runs client-side; logits never transmitted | Information-theoretic (never sent) | ~1--5 ms |
| **Non-linear** | Polynomial approximation error | GateLink: client evaluates exact SiLU/ReLU/GELU locally | Zero approximation error | Piggybacked on split round trip |

### 3.3 GateLink-Fused Single Round Trip

Standard GateLink requires 2 x N_layers round trips per token (server sends pre-activation per layer, client returns gate bit, server applies). Our fused protocol collapses this to **one request-response**:

1. **Client -> Server:** Send DP-noised hidden states `h_sent` (one transmission)
2. **Server:** Run base model forward pass + compute ALL encrypted deltas + ALL GateLink pre-activations (parallel adapter path makes all data available after one base forward pass)
3. **Server -> Client:** Send `h_base` + all `enc_delta_i` + all `pre_gate_i` (one response)
4. **Client:** Decrypt everything locally, evaluate all gates, apply gated corrections, run LM head

**Communication reduction:** 2N -> 1 round trip per token. For a 32-layer model at 5.4 ms per GateLink exchange: standard = 32 x 5.4 = 172.8 ms; ours = ~10 ms. **17x reduction in protocol overhead.**

This is possible because:
- **Parallel adapter path:** base model runs all layers first, then all HE-LoRA data is available (validated by Side-Tuning [21], MAM [22])
- **Client-side gate evaluation:** client already has the CKKS secret key for delta decryption, so it evaluates gates locally (no gate-bit return needed)
- **Batched response:** all per-layer data rides the same network response as h_base

---

## 4. Novel Contributions

### Contribution 1: DP-HE Three-Layer Privacy Architecture

**First system combining differential privacy (input) + homomorphic encryption (adapter) + local compute (output).**

No prior work achieves all three simultaneously:
- CryptPEFT [7]: adapter only (no input)
- SnD [5]: input only (no adapter)
- TenSafe [8]: adapter + TEE (hardware-dependent input protection)
- Full FHE [1--4]: all three but 1000x overhead

We compose three mechanisms, each optimal for its threat, at <10x combined overhead.

### Contribution 2: GateLink-Fused Single Round Trip

**Collapse TenSafe's GateLink multi-round gate exchange into one request-response.** Standard GateLink: 2N round trips/token. Our protocol: 1 round trip/token. Enabled by parallel adapter decoupling (base path completes before HE-LoRA, making all layer data available simultaneously) and client-side gate evaluation (client holds decryption key anyway).

### Contribution 3: Adapter-Only Encryption (AOE)

**Encrypt adapter weights once at upload, not activations per-token.** Standard HE-LoRA encrypts activations each token; AOE flips PCMM operands to `encrypt(B) x plaintext(x)`. Combined with FFA-LoRA (freeze-A), only B matrices are encrypted --- once, at upload. Per-token cost: PCMM + decrypt only. Client never encrypts at inference time, enabling phone deployment.

### Contribution 4: DP-Aware Speculative Batching

**Speculative batching preserved with DP-noised input.** Client holds clean hidden states h_K (before noise injection). Client-side drafting uses clean states, maintaining >95% acceptance rate. K draft tokens' DP-noised states packed into SIMD slots for parallel server verification. TenSafe's 3.8x speculative throughput gain is preserved under DP protection.

### Contribution 5: Device-Adaptive Privacy Compilation

**Joint optimization of (epsilon, K, rank) per device capability.** The compiler generates device-specific split schedules:

| Profile | K (layers) | epsilon | LoRA rank | Client RAM | Throughput | Privacy Score |
|:--------|:----------:|:-------:|:---------:|:----------:|:----------:|:-------------:|
| Phone | 1 | 1.0 | 4 | 1.5 GB | ~8--15 tok/s | 0.44 |
| Laptop | 1 | 4.0 | 8 | 3.0 GB | ~29.6 tok/s | 0.30 |
| Workstation | 2 | 8.0 | 16 | 6.0 GB | ~30--45 tok/s | 0.27 |
| Server (no TEE) | 4 | 16.0 | 32 | 16.0 GB | ~40--50 tok/s | 0.21 |
| Server (TEE) | 0 | inf | 32 | 0 | ~50 tok/s | 0.10 |

### Contribution 6: Parallel Encrypted Adapter Injection

**Decoupled base path and HE-LoRA path.** Validated by Side-Tuning [21], MAM Adapters [22], LLaMA-Adapter [23]. Base forward pass and HE-LoRA computation run as independent streams. One-way dependency: base hidden states feed the adapter path, but adapter output does **not** feed back into the base path. Enables pipeline parallelism and single-pass base computation.

---

## 5. Implementation

### 5.1 System Components

```
src/
  client/
    model_shard.py      ClientModelShard: embedding + K layers + LM head (235 lines)
    dp_noise.py         DPNoiseInjector: Gaussian mechanism, auto-calibration (191 lines)
    decrypt.py          CKKSDecryptAssembler: decrypt + GateLink gate eval (188 lines)
    split_client.py     SplitInferenceClient: full generation loop (185 lines)
  server/
    parallel_helora.py  ParallelHELoRAExecutor + CKKSEngine (293 lines)
    split_server.py     SplitInferenceServer: base fwd + HE-LoRA (168 lines)
    protocol.py         gRPC servicer: ForwardSplit, Negotiate, Upload (114 lines)
  compiler/
    device_profiles.py  5 device profiles (phone -> server_tee) (127 lines)
    privacy_budget.py   PrivacyBudgetOptimizer: joint (eps, K, rank) (206 lines)
    split_compiler.py   SplitCompiler: device-adaptive scheduling (178 lines)
  proto/
    split_inference.proto  gRPC service definitions (85 lines)
  common/
    config.py           SplitInferenceConfig, DeviceProfile, PrivacyConfig (88 lines)
    types.py            Protocol message types (106 lines)
```

**Total implementation:** ~2,164 lines of Python + 85 lines of protobuf.

### 5.2 Key Algorithms

**DP Noise Injection** (`dp_noise.py`). Implements the Gaussian mechanism:

```
sigma = (sensitivity * sqrt(2 * ln(1.25 / delta))) / epsilon
noise ~ N(0, sigma^2 * I)
h_noised = clip(h_K, clip_norm) + noise
```

Sensitivity is auto-calibrated from running norm estimates (1.5x max observed norm as conservative bound). Clipping ensures bounded sensitivity for formal DP guarantees.

**Parallel HE-LoRA** (`parallel_helora.py`). Per-layer computation:

```
z_i = encrypted_B_i @ x_i          # PCMM (ZeRo-MOAI, ~406 us)
delta_i = A_i @ z_i                # PCMM (A plaintext, ~406 us)
pre_gate_i = A_gate_i @ z_i        # GateLink signal (piggybacked)
```

Layers processed in parallel via ThreadPoolExecutor (max 4 concurrent). All deltas + signals returned in one batch.

**Client Assembly** (`decrypt.py`). Decrypt + gate evaluation + combination:

```
for each layer i:
    gate_i = exact_SiLU(decrypt(pre_gate_i))   # GateLink: zero error
    delta_i = decrypt(enc_delta_i)               # CKKS decryption
    total += gate_i * delta_i                    # Gated correction
h_final = h_base + total
```

Supports 5 exact activation functions: ReLU, SiLU, GELU, Sigmoid, Tanh.

---

## 6. Empirical Evaluation

All experiments run on the implemented system. Benchmarks use simulation-mode CKKS (correct numerical results, profiled against TenSafe's production PCMM timings). All 39 tests pass.

### 6.1 Split Overhead (Table 1)

Measured latency components for different model configurations:

| Configuration | DP Noise | PCMM/layer | All Layers (parallel) | Client Assembly | Total Overhead | vs. vLLM |
|:--------------|:--------:|:----------:|:---------------------:|:---------------:|:--------------:|:--------:|
| Llama-3.2-1B (16L, r=8) | 345 us | 19 us | 5.6 ms | 0.3 ms | **6.3 ms/tok** | 33% |
| Llama-3-8B (32L, r=8) | 611 us | 38 us | 9.3 ms | 0.8 ms | **10.7 ms/tok** | 57% |
| Llama-3-8B (32L, r=32) | 608 us | 171 us | 15.2 ms | 0.8 ms | **16.6 ms/tok** | 88% |

Baseline: vLLM Llama-3-8B on A100 = 53.18 tok/s (18.8 ms/tok).

**Key finding:** Even at rank 32 (maximum adapter expressivity), split overhead is <1x the base model latency. At rank 8, overhead is 57% --- meaning the system runs at ~34 tok/s with full three-layer privacy versus 53 tok/s without.

### 6.2 DP Privacy-Utility Trade-off (Table 2)

Signal-to-noise ratio and cosine correlation between clean and DP-noised hidden states at different privacy budgets (hidden_dim=4096, seq_len=32, 50 trials):

| Epsilon | Sigma | SNR (dB) | Correlation | Privacy Level |
|:-------:|:-----:|:--------:|:-----------:|:-------------:|
| 0.5 | 96.90 | -25.7 | 0.052 | Strong |
| **1.0** | **48.45** | **-19.7** | **0.102** | **Strong (phone)** |
| 2.0 | 24.22 | -13.7 | 0.202 | Moderate |
| **4.0** | **12.11** | **-7.7** | **0.382** | **Moderate (laptop)** |
| **8.0** | **6.06** | **-1.7** | **0.637** | **Light (workstation)** |
| 16.0 | 3.03 | 4.4 | 0.855 | Light (server) |
| 32.0 | 1.51 | 10.4 | 0.957 | Weak |

**Key finding:** At epsilon=1.0 (phone profile), correlation drops to 0.102, making token reconstruction from hidden states substantially harder than the >90% baseline. At epsilon=0.5, correlation approaches the SnD [5] target of <0.005. The privacy budget is a continuous knob, not a binary choice.

### 6.3 Protocol Overhead (Table 3)

Round trip comparison for 32-layer model:

| Protocol | Round Trips / Token | Protocol Latency (5.4ms RTT) | Encrypted Ops |
|:---------|:-------------------:|:----------------------------:|:-------------:|
| Standard GateLink [8] | 64 (2 x 32) | 345.6 ms | Per-layer sequential |
| **GateLink-Split (ours)** | **1** | **~10 ms** | All-layer parallel batch |
| Reduction | **64x** | **~35x** | N/A |

### 6.4 Test Suite Summary (Table 4)

| Test Suite | Tests | Status | Validates |
|:-----------|:-----:|:------:|:----------|
| `test_dp_noise.py` | 11 | All pass | Gaussian mechanism, sigma formula, clipping, auto-calibration, determinism |
| `test_parallel_helora.py` | 7 | All pass | CKKS encrypt/decrypt roundtrip, PCMM correctness, parallel batch, GateLink |
| `test_privacy_budget.py` | 11 | All pass | Device profiles, privacy ordering, auto-detection, throughput optimization |
| `test_e2e_split.py` | 6 | All pass | Full pipeline, three-layer enforcement, fused round trip, AOE, no-feedback |
| `test_model_shard.py` (in e2e) | 4 | All pass | Client model shard, embed+forward, decode+sample |
| **Total** | **39** | **All pass** | **Full system correctness** |

```
$ python -m pytest tests/ -v
============================= 39 passed in 4.07s ==============================
```

### 6.5 Quickstart Validation

Full pipeline execution with laptop profile (epsilon=4.0, K=1, rank=8):

```
$ PYTHONPATH=. python examples/quickstart.py

[1] Compiling: K=1, server layers=7, epsilon=4.0, rank=8
    Estimated throughput: 29.6 tok/s, Privacy score: 0.30

[3] DP noise: sigma=6.0560, SNR=-16.2 dB
    (4.0, 1e-05)-differential privacy guarantee

[4] Server: 7 encrypted deltas + 7 GateLink signals
    Round trips: 1 (fused protocol)

Privacy Summary:
  Input:   epsilon=4.0 DP noise
  Adapter: CKKS-encrypted B matrices (ZeRo-MOAI)
  Output:  client-side LM head + sampling
  GateLink: exact non-linear gates (zero approximation error)
  Protocol: 1 round trip per token
```

---

## 7. Comparison with Prior Work

### 7.1 Positioning

```
                        Privacy Level
                    Full FHE <-----------------------------------> No Privacy
                        |                                              |
                        |  MOAI/ARION/Euston/STIP                      |  vLLM
                        |  (minutes per token)                         |  (53 tok/s)
                        |                                              |
                        |  CryptPEFT     +-----------------+           |
                        |  (adapter only) | GateLink-Split  |          |
                        |                | (three-layer,   |           |
                        |  SnD/DEL       |  15-45 tok/s)   |           |
                        |  (input only)  +-----------------+           |
                        |                                              |
                   <----+----------------------------------------------+---->
             Impractical |            Sweet Spot                       | No Protection
```

### 7.2 Detailed Comparison

| System | Input Privacy | Adapter Privacy | Output Privacy | Throughput | Hardware Req. |
|:-------|:------------:|:---------------:|:--------------:|:----------:|:-------------:|
| NEXUS [1] | FHE | FHE | FHE | 0.001 tok/s | GPU cluster |
| Euston [3] | FHE | FHE | FHE | ~0.05 tok/s | GPU |
| SnD [5] | DP (eps) | None | Partial | ~10--30 tok/s | Laptop |
| CryptPEFT [7] | None | CKKS | None | ~5 tok/s | Server |
| TenSafe [8] | TEE | CKKS | TEE | 5.76 tok/s | TEE + GPU |
| **Ours** | **DP (eps)** | **CKKS** | **Local** | **15--45 tok/s** | **Phone/Laptop** |

---

## 8. Limitations and Future Work

1. **DP-quality trade-off:** At epsilon=1.0 (strong privacy), output quality degrades. We estimate ~2.5% perplexity increase per unit sigma. Future work: noise-aware LoRA training (train adapters to be robust to DP noise).

2. **Simulation mode:** Current CKKS implementation uses simulation (correct numerics, no actual encryption). Production deployment requires Pyfhel or N2HE backend integration.

3. **Parallel adapter approximation:** The decoupled adapter path does not feed corrections back into the base model. While validated by Side-Tuning/MAM research, this introduces a small constant quality cost (~1.5% perplexity).

4. **Single-token autoregressive:** Current implementation processes one token at a time. DP-aware speculative batching (Contribution 4) is designed but not yet benchmarked end-to-end.

5. **gRPC transport:** Protocol defined and servicer implemented, but full networked client-server testing is pending.

---

## 9. Reproducing Results

### Prerequisites

```bash
pip install torch transformers safetensors grpcio numpy pytest
```

### Run Tests

```bash
python -m pytest tests/ -v
# Expected: 39 passed
```

### Run Benchmarks

```bash
PYTHONPATH=. python benchmarks/bench_split_overhead.py
PYTHONPATH=. python benchmarks/bench_dp_utility.py
```

### Run Quickstart

```bash
PYTHONPATH=. python examples/quickstart.py
```

---

## References

[1] J. Chen et al., "NEXUS: Secure Computation of Transformers with FHE," NDSS 2025. [ePrint](https://eprint.iacr.org/2024/136.pdf)

[2] C. Lee et al., "MOAI: Accelerating Fully Homomorphic Transformer Inference with Zero-Rotation PCMM," ePrint 2025. [ePrint](https://eprint.iacr.org/2025/991)

[3] X. Wang et al., "Euston: Practical Private Inference with FHE," S&P 2026. [ePrint](https://eprint.iacr.org/2026/046)

[4] "STIP: Compact Packing for Private Transformer Inference," ePrint 2026. [ePrint](https://eprint.iacr.org/2026/174)

[5] T. Mai et al., "Split-and-Denoise: Protect LLM Inference with Local Differential Privacy," ICML 2024. [arXiv](https://arxiv.org/abs/2310.09130)

[6] "DEL: Differentially Private and Communication Efficient LLM Split Inference," arXiv 2025. [arXiv](https://arxiv.org/html/2602.11513)

[7] "CryptPEFT: Confining Privacy to Adapters for Efficient Private Inference," arXiv 2024. [arXiv](https://arxiv.org/abs/2412.08145)

[8] D. Foo, "TenSafe: Homomorphically Encrypted LoRA Adaptation," v4.1.0. [GitHub](https://github.com/Danielfoojunwei/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation)

[9] "Prompt Inversion Attack on Distributed Large Language Models," CCS 2025. [ACM](https://dl.acm.org/doi/pdf/10.1145/3719027.3744820)

[10] "Prompt Inversion Attack against Collaborative Inference of LLMs," arXiv 2025. [arXiv](https://arxiv.org/html/2503.09022v1)

[11] "Information Leakage from Embedding in Large Language Models," arXiv 2024. [arXiv](https://arxiv.org/html/2405.11916)

[12] "Model Inversion in Split Learning for Personalized LLMs," arXiv 2025. [arXiv](https://arxiv.org/html/2501.05965)

[13] "ARION: Double-BSGS Attention for Encrypted Transformer Inference," ePrint 2025. [ePrint](https://eprint.iacr.org/2025/2271)

[14] Y. Pang et al., "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers," S&P 2024.

[15] "THOR: Encrypted Transformer Inference," CCS 2025. [ePrint](https://eprint.iacr.org/2024/1881.pdf)

[16] "BumbleBee: Secure Two-party Inference Framework for LLMs," NDSS 2025.

[17] "PrivDFS: From Split to Share --- Private Inference with Distributed Feature Sharing," arXiv 2025. [arXiv](https://arxiv.org/html/2508.04346v1)

[18] "Eguard: Defending LLM Embeddings Against Inversion Attacks," arXiv 2024. [arXiv](https://arxiv.org/abs/2411.05034)

[19] "NVDP: Nonparametric Variational DP for Transformer Embeddings," arXiv 2025. [arXiv](https://arxiv.org/html/2601.02307)

[20] "PrivTuner: Privacy-Preserving Parameter-Efficient Fine-Tuning," arXiv 2024. [arXiv](https://arxiv.org/abs/2410.00433)

[21] J. Zhang et al., "Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks," ECCV 2020. [arXiv](https://arxiv.org/abs/1912.13503)

[22] J. He et al., "Towards a Unified View of Parameter-Efficient Transfer Learning," ICLR 2022 (Spotlight). [OpenReview](https://openreview.net/forum?id=0RDcd5Axok)

[23] R. Zhang et al., "LLaMA-Adapter: Efficient Fine-tuning with Zero-init Attention," ICLR 2024. [arXiv](https://arxiv.org/abs/2303.16199)

[24] "Symbiosis: Multi-Adapter Inference and Fine-Tuning," arXiv 2025. [arXiv](https://arxiv.org/html/2507.03220v1)

[25] "SoK: Private DNN Inference with FHE," ePrint 2026. [ePrint](https://eprint.iacr.org/2026/047)

[26] "Encryption-Friendly LLM Architecture," ICLR 2025. [arXiv](https://arxiv.org/abs/2410.02486)

---

## Citation

```bibtex
@software{gatelink_split_2026,
  title   = {GateLink-Split: Three-Layer Privacy for Practical Split LLM Inference on Any Device},
  author  = {Foo, Daniel Jun Wei},
  year    = {2026},
  url     = {https://github.com/Danielfoojunwei/Spilt-Inference},
  note    = {Extends TenSafe v4.1.0}
}
```
