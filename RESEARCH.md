# HE Split Inference — Research, Novelty & Architecture Plan

## Building on TenSafe

This work extends [TenSafe](https://github.com/Danielfoojunwei/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation) (v4.1.0), which already provides:
- **HE-LoRA microkernel** — MOAI zero-rotation CKKS compiler + runtime for encrypted LoRA deltas
- **N2HE CUDA backend** — pybind11 GPU backend (stub, ready for implementation)
- **Hybrid CKKS+TFHE** — Gated LoRA with LUT-based activations via programmable bootstrapping
- **Client-server architecture** — HAS (gRPC HE service) + MSS (FastAPI model serving) + vLLM integration
- **5.76 tok/s** on Llama-3-8B (A100-80GB) with HE-LoRA, vs 2.22 tok/s vanilla HE baseline
- **FFA-LoRA** — Freeze-A optimization (50% comm reduction)
- Cost-budgeted compiler (<=16 rotations/token, <=64/layer)

---

## 1. Why Our Split Inference is Novel

### 1.1 The Fundamental Inversion: Encrypt the Personalization, Not the Computation

**Every SOTA system encrypts the wrong thing.**

The entire field — MOAI, ARION, Euston, BOLT, BLB, THOR, BumbleBee — assumes this threat model:

```
SOTA: Client has private INPUT → encrypt input → server runs ENTIRE model on ciphertext
      Result: BERT-base takes 10-18 minutes. LLaMA-3-8B is impractical.
```

They encrypt the **full forward pass**. Every attention head, every FFN, every Softmax, every LayerNorm — all on ciphertext. This is why:
- NEXUS: 1103 seconds for BERT-base (128 tokens)
- THOR: 10 minutes on a single GPU
- BumbleBee: 8.2 minutes + 25.3 GB communication for ONE token of GPT-2
- Euston (SOTA 2026): Still minutes-scale

**Our approach inverts this entirely:**

```
TenSafe Split: Base model runs in PLAINTEXT (fast, normal speed)
               Only LoRA adapter deltas run under CKKS (tiny, microsecond-scale)
               Client handles embedding + sampling (server never sees raw text)
               Result: 5.76 tok/s TODAY. Target: near-plaintext speed.
```

The insight: **The base model is public. The personalization is private.** So encrypt only the personalization.

### 1.2 What This Means Concretely

| Aspect | SOTA (Full Encrypted Inference) | TenSafe Split Inference |
|--------|-------------------------------|------------------------|
| **What's encrypted** | Entire hidden state (d=4096) through every layer | Only LoRA delta (rank 8-32) per layer |
| **Matrix ops on ciphertext** | QKV projections, FFN, attention scores — all CCMM/PCMM | 2 small PCMM per layer (B@x, A@Bx) |
| **Nonlinear on ciphertext** | Softmax, GELU, LayerNorm — the hard part | None (base model handles these in plaintext) |
| **Bootstrapping needed** | Yes, frequently (depth exhausted every few layers) | No (depth 2-3 sufficient for LoRA) |
| **CKKS parameters** | N=65536, deep modulus chains, massive ciphertexts | N=16384, shallow chain, small ciphertexts |
| **Per-token latency** | Seconds to minutes | ~812 microseconds (benchmarked) |
| **Communication** | 15-60 GB per inference (BERT-base) | ~1 MB encrypted adapter weights |
| **Rotations** | Thousands per inference (MOAI removes 2448, still has hundreds) | <=16 per token (budget-enforced) |

### 1.3 The Three-Way Privacy Split

```
┌─────────────────────────────────────────────────┐
│                   CLIENT                         │
│                                                  │
│  [1] Tokenize + Embed (PRIVATE — raw text)       │
│  [2] Run layer 0 (optional warmup)               │
│           │                                      │
│           ▼ send hidden states to server          │
│           │ (these are base-model representations │
│           │  not raw text — reduced privacy risk) │
│                                                  │
│  [5] ◄── receive base output + encrypted delta   │
│  [6] Decrypt LoRA delta, add to base output      │
│  [7] Run LM Head + Sampling (PRIVATE — logits)   │
│           │                                      │
│           ▼ next token (loop)                    │
└─────────────────────────────────────────────────┘
                    │          ▲
                    ▼          │
┌─────────────────────────────────────────────────┐
│                   SERVER                         │
│                                                  │
│  [3] Run transformer layers 1..N in PLAINTEXT    │
│      (base model — public weights, normal speed) │
│                                                  │
│  [4] For each layer, compute HE-LoRA delta:      │
│      ├── Encrypt activations (CKKS)              │
│      ├── B @ x (PCMM, zero-rotation)             │
│      ├── A @ Bx (PCMM, zero-rotation)            │
│      ├── Rescale                                  │
│      └── Inject encrypted delta into output      │
│                                                  │
│  Server NEVER sees:                              │
│    ✗ Raw text (client embeds locally)             │
│    ✗ Adapter weights (encrypted under CKKS)       │
│    ✗ Final logits (client decodes locally)        │
│    ✗ Sampling distribution (client-side)          │
│                                                  │
│  Server DOES see:                                │
│    ✓ Intermediate hidden states (base model)      │
│    ✓ Base model weights (public/licensed)         │
└─────────────────────────────────────────────────┘
```

**Privacy guarantee:** Server learns the base model's representation of the input (which leaks some info about the input domain), but never learns:
- The raw text
- The user's personalized adapter (their private fine-tuning data is protected)
- The final output distribution (what the model actually says)

This is a **pragmatic privacy model** — not perfect information-theoretic privacy, but enormously practical.

### 1.4 Why No One Else is Doing This

1. **Academic incentive mismatch**: Papers get published by solving the hard crypto problem (full encrypted inference). "Just encrypt the adapter" seems too simple for a top-tier venue.

2. **Threat model assumption**: Most papers assume the server is adversarial with respect to the INPUT. We assume the server is adversarial with respect to the PERSONALIZATION. Different threat, different solution.

3. **LoRA wasn't mainstream until 2023-2024**: The older HE inference papers (Iron 2022, BOLT 2024) predate the LoRA explosion. They didn't have "encrypt only the adapter" as an option.

4. **The split was missing**: Even papers that use LoRA+HE (PrivTuner, Encryption-Friendly LLM) don't combine it with a client-side embedding/sampling split. They still encrypt the full input.

---

## 2. New Directions: Efficient, Cheap, Any-Device Split Inference

### 2.1 Direction 1: "Adapter-Only Encryption" (AOE)

**Core thesis:** Don't encrypt hidden states at all. Encrypt ONLY the adapter weights.

```
Current TenSafe:  encrypt(activations) → PCMM with plaintext weights → decrypt(delta)
Proposed AOE:     plaintext activations → PCMM with encrypt(adapter weights) → encrypted delta → client decrypts
```

This flips PCMM: instead of `plaintext_weight × ciphertext_activation`, do `ciphertext_weight × plaintext_activation`. Both are PCMM — same cost, same zero-rotation guarantee. But now:

- **No per-token encryption/decryption on client** — adapter weights encrypted once at upload
- **Server stores encrypted adapters** — serves them to any request
- **Client only decrypts the final delta** — one small decryption per layer per token
- **Adapter weights are static** — encrypt once, use forever (amortized cost → zero)

**Why this is huge for any-device:**
- Client doesn't need to encrypt anything at inference time
- Client only needs CKKS decryption capability (much lighter than encryption)
- A phone can be the client — just decrypt small deltas

### 2.2 Direction 2: Progressive Privacy Tiers

Not every user needs the same privacy level. Offer tiers:

| Tier | Client Runs | Encrypted | Overhead | Device |
|------|------------|-----------|----------|--------|
| **Tier 0**: No privacy | Nothing | Nothing | 0% | Any (API call) |
| **Tier 1**: Adapter privacy | Nothing extra | LoRA weights only (AOE) | ~2-5% | Phone/laptop |
| **Tier 2**: Input+Adapter privacy | Embed + LM head | LoRA weights + embed split | ~10-15% | Laptop |
| **Tier 3**: Full split | Embed + first/last N layers + sampling | LoRA + hidden states at split boundary | ~50-90% | Workstation |
| **Tier 4**: Full encrypted | Everything | Full inference under FHE | 100-1000x | Research only |

**Users choose their tier based on their device and threat model.** The compiler generates different schedules for each tier. This is a product insight, not just a research one.

### 2.3 Direction 3: Streaming Delta Injection (SDI)

Instead of one big encrypted round-trip, **stream tiny encrypted corrections alongside the plaintext forward pass:**

```
Layer 1: base_output_1 = transformer_layer_1(x)        # plaintext, fast
         delta_1 = HE_LoRA(x, encrypted_A1, encrypted_B1)  # CKKS, parallel
         output_1 = base_output_1 + decrypt(delta_1)

Layer 2: base_output_2 = transformer_layer_2(output_1)  # plaintext, fast
         delta_2 = HE_LoRA(output_1, encrypted_A2, encrypted_B2)  # CKKS, parallel
         ...
```

The HE computation for layer N can overlap with the plaintext computation for layer N+1. This is **pipeline parallelism for HE** — the LoRA deltas are computed in a parallel stream and injected as they become ready.

**Latency impact:** If HE-LoRA per layer takes ~812 us and base layer takes ~500 us, the HE computation is nearly hidden behind the base forward pass. Total overhead approaches the single-layer HE cost, not the sum.

### 2.4 Direction 4: Encrypted Adapter Marketplace

If adapter weights are encrypted and the server never sees them:

- **Users can upload private adapters** fine-tuned on their proprietary data
- **Multiple users share the same base model** (server-side) with different encrypted adapters
- **Adapters become portable, private assets** — users can switch providers without exposing their fine-tuning
- **No adapter theft possible** — server cannot extract adapter weights from ciphertexts

This is a **business model**, not just a privacy feature. Think "bring your own encrypted adapter" to any cloud LLM provider.

### 2.5 Direction 5: Client-Adaptive Compilation

TenSafe's compiler already produces cost-budgeted schedules. Extend it:

```python
# Compiler profiles based on client capability
profiles = {
    "phone":       {"max_rotations": 4,  "max_depth": 1, "tier": "AOE"},
    "laptop":      {"max_rotations": 8,  "max_depth": 2, "tier": "split"},
    "workstation": {"max_rotations": 16, "max_depth": 3, "tier": "full_split"},
    "server":      {"max_rotations": 64, "max_depth": 5, "tier": "research"},
}
```

The client announces its capability → server selects the pre-compiled schedule → adapts the split point, encryption parameters, and LoRA rank dynamically.

**Key insight:** LoRA rank can be reduced (rank 32 → rank 4) to fit weaker devices. Lower rank = fewer HE operations = faster. The quality loss from rank reduction is often acceptable.

### 2.6 Direction 6: FFA-LoRA + Quantized HE

TenSafe already implements FFA-LoRA (Freeze-A). Push further:

- **A matrix**: public, plaintext, frozen — zero HE cost
- **B matrix**: encrypted — but quantize to 4-bit before encrypting
- **Effect**: 4-bit B matrix packs 8x more values per CKKS slot (8192 slots → 65K values)
- **Combined**: 50% reduction (FFA) × 8x packing (quant) = **16x less HE work**

At 16x reduction, the ~812 us per-layer HE cost drops to ~50 us — completely hidden behind base model latency. **HE overhead effectively disappears.**

---

## 3. SOTA Landscape (Context for Positioning)

### 3.1 Pure FHE — What We're NOT Doing (and Why)

| System | Venue | Latency | Why We Diverge |
|--------|-------|---------|----------------|
| [NEXUS](https://eprint.iacr.org/2024/136.pdf) | NDSS 2025 | ~1103s (CPU) | Encrypts entire model — impractical |
| [MOAI](https://eprint.iacr.org/2025/991) | ePrint 2025 | Best GPU pure FHE | We use MOAI's zero-rotation technique, but only for LoRA delta, not full model |
| [ARION](https://eprint.iacr.org/2025/2271) | ePrint 2025 | 2.5x faster than MOAI | Double-BSGS useful if we ever need encrypted attention |
| [Euston](https://eprint.iacr.org/2026/046) | S&P 2026 | SOTA pure FHE | SVD decomposition idea applicable to LoRA weight encoding |
| [STIP](https://eprint.iacr.org/2026/174) | ePrint 2026 | 1.6x faster than Euston | Compact packing ideas reusable |

**We borrow their techniques (zero-rotation, column packing, SIMD batching) but apply them to a 100-1000x smaller problem (LoRA delta vs full forward pass).**

### 3.2 Hybrid HE+MPC — Relevant for Gated LoRA Path

| System | Venue | Relevance to Us |
|--------|-------|----------------|
| [BLB](https://eprint.iacr.org/2025/1532) | USENIX Security 2025 | CKKS↔MPC bridge protocol — useful for our CKKS↔TFHE bridge |
| [CryptoGen](https://arxiv.org/abs/2602.08798) | arXiv 2025 | Encrypted KV-cache reuse — directly applicable to our autoregressive loop |
| [Cachemir](https://arxiv.org/abs/2602.11470) | arXiv 2025 | FHE-only KV cache — alternative approach for our pipeline |
| [BOLT](https://www.semanticscholar.org/paper/BOLT:-Privacy-Preserving,-Accurate-and-Efficient-Pang-Zhu/0f7bbe9837026560a934de8a74d233678bd55f57) | S&P 2024 | Baseline everyone beats |
| [BumbleBee](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf) | NDSS 2025 | Ciphertext interleaving — applicable to our batch packing |

### 3.3 NTU DTC / HintSight Lineage — Our Direct Ancestors

| Paper | What We Inherit |
|-------|----------------|
| [Hybrid PP-NN (NTU/HintSight, TDSC 2024)](https://eprint.iacr.org/2023/647) | **Split DNN architecture** — plaintext open network (client) + ciphertext private network (server). The foundational idea. |
| [MOAI (NTU DTC, ePrint 2025)](https://eprint.iacr.org/2025/991) | **Zero-rotation PCMM** — column packing, rotation-free Softmax/LayerNorm. TenSafe's compiler already implements this. |
| [PrivTuner (NTU, arXiv 2024)](https://arxiv.org/abs/2410.00433) | **FHE + LoRA** — privacy-preserving parameter-efficient fine-tuning. Direct predecessor to TenSafe's HE-LoRA. |
| [NTU Hybrid PP-NN portal](https://www.ntu.edu.sg/innovates/tech-portal/tech-offers/detail/new-model-hybrid-privacy-preserving-neural-networks) | **HintSight commercialization** — LUT-based nonlinear evaluation, <1s inference. |

**N2HE context:** TenSafe already has `scripts/n2he/` (build scripts) and `src/he_lora_microkernel/n2he/` (backend module) — N2HE is TenSafe's **own native CUDA HE backend**, not a separate paper. It stands for the Native-Node Homomorphic Encryption GPU kernel library used by the HE-LoRA microkernel.

### 3.4 Architecture-Level Innovations We Can Leverage

| Paper | Technique | How We Use It |
|-------|-----------|---------------|
| [Encryption-Friendly LLM (ICLR 2025)](https://arxiv.org/abs/2410.02486) | Gaussian kernel attention (replace softmax), LoRA avoids CCMM | If we ever need encrypted attention (Tier 3+), use GK instead of polynomial softmax |
| [StriaNet (arXiv 2025)](https://arxiv.org/abs/2601.21287) | ExRot-Free convolution, Cross Kernel | Applicable to conv-based adapter layers if we add them |
| [CryptPEFT](https://arxiv.org/abs/2412.08145) | Confine privacy to adapters only | Validates our core thesis — 20-291x speedup over encrypting everything |
| [Split HE (arXiv 2022)](https://arxiv.org/abs/2202.13351) | SplitNN + TFHE, 3-part model split | Original split idea, but with full encryption — we do it lighter |

---

## 4. TenSafe Components We Build On

### 4.1 Already Implemented (in TenSafe)

| Component | Location | Status |
|-----------|----------|--------|
| CKKS MOAI backend | `src/crypto_backend/ckks_moai/` | Working (Pyfhel) |
| LoRA IR compiler | `src/he_lora_microkernel/compiler/` | Working (20+ op types) |
| MOAI zero-rotation scheduler | `compiler/scheduler.py` | Working |
| Cost model / budget enforcement | `compiler/cost_model.py` | Working (<=16 rot/tok) |
| SIMD batch-first packer | `compiler/packer.py` | Working |
| Token-by-token executor | `runtime/executor.py` | Working |
| Gated LoRA (CKKS+TFHE) | `hybrid_compiler/gated_lora/` | Working |
| CKKS↔TFHE bridge | `hybrid_compiler/bridge/` | Working |
| TFHE LUT activations | `hybrid_compiler/tfhe_lut/` | Working (ReLU, GELU, Sigmoid) |
| HAS gRPC server | `services/has/` | Working |
| MSS REST API | `services/mss/` | Working |
| vLLM adapter | `backend/vllm_adapter/` | Working (hook-based delta injection) |
| TenSafe Python client | `client/client.py` | Working |
| TypeScript SDK | `sdk/typescript/` | Working |
| N2HE CUDA backend | `he_lora_microkernel/n2he/` | Stub (architecture defined) |
| GPU CKKS backend | `backend/gpu_ckks_backend.py` | Stub (SimulationBackend working) |

### 4.2 What Split Inference Adds

| New Component | Purpose | Builds On |
|---------------|---------|-----------|
| **Client-side embedding layer** | Tokenize + embed locally, never send raw text | TenSafe client SDK |
| **Client-side LM head + sampler** | Decode logits locally, server never sees output distribution | New |
| **Split-point negotiation** | Client announces capability → server adapts split | HAS protocol extension |
| **Streaming delta injection** | Pipeline HE-LoRA parallel to base forward pass | Runtime executor extension |
| **Adapter-Only Encryption mode** | Encrypt adapter weights at upload, not per-token | Compiler + key manager |
| **Progressive privacy tiers** | Tier 0-4 compilation profiles | Cost model extension |
| **Encrypted KV-cache** | Reuse LoRA delta KV contributions across tokens | New (CryptoGen-inspired) |
| **Client-adaptive compiler** | Device-specific schedule generation | Compiler profiles extension |

---

## 5. Implementation Plan

### Phase 1: Client-Side Embedding & Sampling Split
- [ ] Extract embedding layer + LM head from target model (Llama-3-8B)
- [ ] Package as lightweight client runtime (PyTorch inference-only, no training)
- [ ] Client sends post-embedding hidden states to server via existing HAS gRPC
- [ ] Client receives pre-LM-head hidden states + decrypted LoRA delta
- [ ] Client runs LM head + sampling locally
- [ ] Benchmark: measure overhead of split vs. current TenSafe flow
- [ ] Validate: server never sees raw tokens or output logits

### Phase 2: Adapter-Only Encryption (AOE)
- [ ] Implement "encrypt weights, not activations" mode in compiler
- [ ] Modify PCMM to `ciphertext_weight × plaintext_activation`
- [ ] Adapter weights encrypted once at upload (amortized cost)
- [ ] Client only needs decryption at inference time
- [ ] Benchmark: compare per-token latency vs current encrypt-activations mode
- [ ] Target: <100 us overhead per layer (hidden behind base model latency)

### Phase 3: Streaming Delta Injection (Pipeline Parallelism)
- [ ] Modify runtime executor to overlap HE-LoRA with base forward pass
- [ ] Layer N HE-LoRA runs concurrently with layer N+1 base forward
- [ ] Implement async delta injection into vLLM adapter hooks
- [ ] Benchmark: measure effective overhead (should approach single-layer HE cost)

### Phase 4: Progressive Privacy Tiers
- [ ] Define tier profiles (phone/laptop/workstation/server)
- [ ] Compiler generates tier-specific schedules
- [ ] Client capability negotiation in HAS protocol
- [ ] Adaptive LoRA rank reduction for weaker devices
- [ ] FFA-LoRA + quantized HE integration (16x reduction path)

### Phase 5: N2HE GPU Backend (Production Performance)
- [ ] Implement N2HE CUDA kernels (currently stub)
- [ ] Zero-rotation PCMM on GPU
- [ ] Batched encrypt/decrypt for streaming
- [ ] Benchmark on A100/H200 vs Pyfhel CPU baseline
- [ ] Target: 10x+ speedup over CPU backend

---

## 6. Competitive Positioning

```
                        Privacy Level
                    Full FHE ◄──────────────────► No Privacy
                        │                              │
                        │  MOAI/ARION/Euston            │  Standard vLLM
                        │  (minutes per token)          │  (53 tok/s)
                        │                              │
                        │         ┌──────────┐         │
                        │         │ TenSafe  │         │
                        │         │  Split   │         │
                        │         │ Inference│         │
                        │         └──────────┘         │
                        │    (5-50 tok/s, practical     │
                        │     privacy, any device)      │
                        │                              │
                   ◄────┼──────────────────────────────┼────►
              Impractical│      Sweet Spot              │ No Protection
                        │                              │
```

**Our position:** We sacrifice theoretical perfect privacy (the server sees base-model hidden states) for **1000x better performance**. The privacy we DO provide — protecting the adapter (user's private fine-tuning) and raw text/output — covers the practical threat model for most real-world deployments.

---

## 7. Key References

### Our Direct Lineage
1. [TenSafe — HE-LoRA Platform](https://github.com/Danielfoojunwei/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation)
2. [MOAI — Zero-rotation (NTU DTC)](https://eprint.iacr.org/2025/991)
3. [Hybrid PP-NN (NTU/HintSight)](https://eprint.iacr.org/2023/647)
4. [PrivTuner — FHE + LoRA (NTU)](https://arxiv.org/abs/2410.00433)

### Techniques We Borrow
5. [ARION — Double-BSGS attention](https://eprint.iacr.org/2025/2271)
6. [Euston — SVD-based HMM (S&P 2026)](https://eprint.iacr.org/2026/046)
7. [CryptoGen — Encrypted KV-cache](https://arxiv.org/abs/2602.08798)
8. [Encryption-Friendly LLM (ICLR 2025)](https://arxiv.org/abs/2410.02486)
9. [CryptPEFT — Adapter-only privacy](https://arxiv.org/abs/2412.08145)
10. [BLB — CKKS↔MPC bridge](https://eprint.iacr.org/2025/1532)

### The Competition (Full Encrypted Inference)
11. [NEXUS (NDSS 2025)](https://eprint.iacr.org/2024/136.pdf)
12. [THOR (CCS 2025)](https://eprint.iacr.org/2024/1881.pdf)
13. [BOLT (S&P 2024)](https://www.semanticscholar.org/paper/BOLT:-Privacy-Preserving,-Accurate-and-Efficient-Pang-Zhu/0f7bbe9837026560a934de8a74d233678bd55f57)
14. [BumbleBee (NDSS 2025)](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)
15. [STIP (ePrint 2026)](https://eprint.iacr.org/2026/174)
16. [Cachemir (arXiv 2025)](https://arxiv.org/abs/2602.11470)

### Architecture Design
17. [StriaNet — Zero-rotation NN design](https://arxiv.org/abs/2601.21287)
18. [Split HE (arXiv 2022)](https://arxiv.org/abs/2202.13351)
19. [Iron (NeurIPS 2022)](https://papers.neurips.cc/paper_files/paper/2022/file/64e2449d74f84e5b1a5c96ba7b3d308e-Paper-Conference.pdf)

### Surveys
20. [Survey on Private Transformer Inference](https://arxiv.org/abs/2412.08145)
21. [SoK: Private DNN Inference with FHE (2026)](https://eprint.iacr.org/2026/047)

### GPU Libraries
22. [Phantom — CUDA HE](https://github.com/encryptorion-lab/phantom-fhe)
23. [FIDESlib — CKKS on GPU](https://arxiv.org/abs/2507.04775)
24. [Lattigo — Go HE](https://github.com/tuneinsight/lattigo)
