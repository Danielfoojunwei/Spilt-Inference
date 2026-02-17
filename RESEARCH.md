# HE Private Inference — SOTA Research & Architecture Plan

## 1. Problem Statement

Run split transformer inference where:
- **Client** tokenizes, embeds, and encrypts hidden states (CKKS) after layer 1
- **Server** runs the bulk of transformer layers on ciphertexts
- **Client** decrypts and runs final layers + sampling locally
- Server **never sees raw text** — pure math, no hardware trust required

**Core bottleneck:** Running 28+ transformer layers on CKKS ciphertexts. Even with zero-rotation optimizations (MOAI), this is slower than TEE. Target: 2–10 tok/s.

---

## 2. SOTA Landscape (as of early 2026)

### 2.1 Pure FHE (Non-Interactive) Approaches

| System | Venue | BERT-base (128 tok) Latency | Key Innovation |
|--------|-------|----------------------------|----------------|
| **NEXUS** | NDSS 2025 | ~1103s (CPU) | RNS-CKKS, SIMD compression, first non-interactive |
| **MOAI** | ePrint 2025/991 | ~52.8% faster than THOR (GPU) | Zero-rotation Softmax/LayerNorm, column packing, interleaved batching |
| **THOR** | ACM CCS 2025 | ~10 min (single GPU) | Diagonal-major encoding, compact packing, adaptive nonlinear approx |
| **ARION** | ePrint 2025/2271 | 2.5× faster than MOAI (e2e) | Double Baby-Step Giant-Step for attention, 82.5% rotation reduction over MOAI |
| **Euston** | IEEE S&P 2026 | 3100× lower user preprocessing, 8.8× faster than NEXUS | SVD-based batched HMM, column-packed HNE, offline-online paradigm |
| **STIP** | ePrint 2026/174 | ~1.6× faster than Euston | Compact packing, follow-up to Euston |

**Current SOTA for pure FHE:** Euston → STIP pipeline (S&P 2026 + follow-up).

### 2.2 Hybrid HE + MPC Approaches

| System | Venue | Key Innovation |
|--------|-------|----------------|
| **BOLT** | IEEE S&P 2024 | HE for linear, MPC for nonlinear; 59.6 GB comm for BERT-base |
| **BumbleBee** | NDSS 2025 | Ciphertext interleaving, 15.5 GB for BERT-large |
| **BLB** | USENIX Security 2025 | First CKKS↔MPC conversion protocol, fine-grained operator fusion; 21× less comm than BOLT |
| **CryptoGen** | arXiv 2602.08798 | First encrypted KV-cache reuse for autoregressive generation; 4.4–7.6× lower per-token latency |
| **Cachemir** | arXiv 2602.11470 | FHE-only generative inference with KV cache |

**Current SOTA for hybrid:** BLB (USENIX Security 2025) for discriminative; CryptoGen for generative.

### 2.3 Split / Hybrid PP-NN Approaches (NTU DTC-aligned)

| System | Venue | Key Innovation |
|--------|-------|----------------|
| **Hybrid PP-NN** (NTU/HintSight) | IEEE TDSC 2024 | Split DNN into plaintext (client) + ciphertext (server) parts; <1s facial recognition vs >1 day for full FHE |
| **MOAI** (NTU DTC authors) | ePrint 2025/991 | Linru Zhang, Kwok Yan Lam et al.; rotation-free Softmax/LayerNorm; Phantom GPU lib; tested on LLaMA-3-8B |
| **GuardML** | ACM SAC 2024 | Hybrid HE using PASTA symmetric cipher + BFV wrapper; 300× bandwidth reduction |
| **PrivTuner** (NTU) | arXiv 2410.00433 | FHE + LoRA for P3EFT (privacy-preserving parameter-efficient fine-tuning) |
| **Encryption-Friendly LLM** | ICLR 2025 | LoRA + Gaussian kernel attention (replace softmax); 6.94× speedup fine-tuning, 2.3× inference |
| **Split HE** | arXiv 2022 | SplitNN + TFHE; model split into 3 parts (client-server-client) |

### 2.4 HE-Friendly Architecture Design

| System | Key Innovation |
|--------|----------------|
| **StriaNet** (2025) | StriaBlock: ExRot-Free convolution + Cross Kernel; eliminates external rotations, 19% internal rotations; 9.78× speedup over VGG |
| **Encryption-Friendly LLM** (ICLR 2025) | Replace softmax with Gaussian kernels; LoRA to avoid CCMM; CCMM is ~10× slower than PCMM |
| **CryptPEFT** | Confine private computation to LoRA adapters; 20–291× speedup over global MPC |

---

## 3. Key Technical Insights

### 3.1 The Rotation Problem
- **Rotation** (ciphertext slot permutation via key switching) is the most expensive CKKS operation
- In BERT-base inference, MOAI removes **2,448 HE rotations**
- ARION further reduces rotations by **82.5%** over MOAI using double-BSGS
- Column packing achieves rotation-free plaintext-ciphertext matrix multiplication (PCMM)
- **Takeaway:** Architecture must minimize rotations — this is the #1 performance lever

### 3.2 Linear vs Nonlinear Split
- **Linear layers** (attention QKV projections, FFN): Well-suited for HE (PCMM)
- **Nonlinear layers** (Softmax, GELU, LayerNorm): The hard part
  - Polynomial approximation (MOAI, THOR): Higher accuracy, more depth budget
  - Interactive MPC (BOLT, BLB): Lower compute, requires communication rounds
  - Lookup tables (HintSight/NTU): Fast but limited precision
- **Takeaway:** For non-interactive split inference, polynomial approximation is necessary

### 3.3 CKKS Depth Budget & Bootstrapping
- CKKS ciphertexts have a fixed multiplicative depth (level budget)
- Once exhausted, **bootstrapping** is needed (~100× slower than other ops)
- Euston uses depth regulation strategies to minimize bootstrapping
- **Takeaway:** Layer count directly impacts bootstrapping frequency → split point matters

### 3.4 PCMM vs CCMM
- **PCMM** (plaintext-ciphertext): Server weights are plaintext, client data is ciphertext → fast
- **CCMM** (ciphertext-ciphertext): Both encrypted → ~10× slower
- LoRA converts most CCMM to PCMM (frozen weights stay plaintext)
- **Takeaway:** Keep model weights in plaintext on server; only client data is encrypted

### 3.5 KV-Cache for Autoregressive Generation
- CryptoGen: First encrypted KV-cache reuse → near-linear latency scaling
- Without KV-cache reuse: quadratic latency growth (re-process entire prefix)
- **Takeaway:** For generative LLMs, KV-cache management is essential

---

## 4. GPU Libraries for CKKS

| Library | Language | GPU | Bootstrapping | Notes |
|---------|----------|-----|---------------|-------|
| **Phantom** | C++ (CUDA) | Yes (A100, H200, 4090) | No (planned) | Used by MOAI, NEXUS; GPLv3; current best perf |
| **FIDESlib** | C++ (CUDA) | Yes | Yes (227.8× over OpenFHE) | Newer; outperforms Phantom in scalability |
| **Cheddar** | C++ (CUDA) | Yes | Yes | Swift FHE for CUDA |
| **Lattigo** | Go | No (CPU) | Yes | Used by ARION; mature; good for prototyping |
| **OpenFHE** | C++ | No (CPU) | Yes | Reference implementation; slower |
| **SEAL/TenSEAL** | C++/Python | No (CPU) | Limited | Microsoft; good Python bindings; used by Split HE |

**Recommendation:** Start with **Lattigo** (Go, CPU) for prototyping, then move to **Phantom** or **FIDESlib** (CUDA) for production GPU perf.

---

## 5. NTU DTC-Aligned Papers (Prof. Kwok Yan Lam's Group)

The NTU Digital Trust Centre, led by Prof. Kwok Yan Lam, has a direct research lineage:

1. **Efficient FHE-based PE-NN** (TDSC 2024) — Split DNN into open (plaintext, client-side) + private (ciphertext, server-side) networks via transfer learning. <1s inference for facial recognition on 16-core CPU.

2. **MOAI** (ePrint 2025/991) — Linru Zhang, Xiangning Wang, Kwok Yan Lam et al. Zero-rotation Softmax/LayerNorm for BERT-base/LLaMA-3-8B. Uses Phantom GPU library. **This is the most directly relevant paper to our architecture.**

3. **PrivTuner** (arXiv 2024) — FHE + LoRA for privacy-preserving fine-tuning.

4. **GuardML** (SAC 2024) — Hybrid HE for resource-constrained devices.

**Note:** I could not locate a specific paper titled "N2HE" from NTU DTC. The closest match is "N2HE" in hypergraph neural networks (Node-to-Hyperedge), which is unrelated to HE cryptography. The user may be referring to the **NTU PE-NN** split model or **MOAI** zero-rotation work. Clarification recommended.

---

## 6. Proposed Architecture: Split HE Inference

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENT                                 │
│                                                               │
│  Input Text                                                   │
│     │                                                         │
│     ▼                                                         │
│  Tokenizer + Embedding Layer (plaintext)                      │
│     │                                                         │
│     ▼                                                         │
│  Layer 0 (optional: 1-2 warmup layers, plaintext)             │
│     │                                                         │
│     ▼                                                         │
│  CKKS Encrypt(hidden_states)  ────────────────► Server        │
│                                                               │
│  ◄──────────────────────────────  CKKS ciphertext result      │
│     │                                                         │
│     ▼                                                         │
│  CKKS Decrypt                                                 │
│     │                                                         │
│     ▼                                                         │
│  Final Layer(s) + LM Head (plaintext)                         │
│     │                                                         │
│     ▼                                                         │
│  Sampling (top-k/top-p) → next token                          │
│                                                               │
│  [Loop: re-encrypt, send for next token]                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                        SERVER                                 │
│                                                               │
│  Receive CKKS ciphertext                                      │
│     │                                                         │
│     ▼                                                         │
│  Transformer Layers 1..N-1 (on ciphertext)                    │
│     ├── Attention: Q,K,V projections (PCMM)                  │
│     │   ├── QK^T (CCMM — the expensive part)                 │
│     │   ├── Softmax (polynomial approx, rotation-free)        │
│     │   └── Attn × V (CCMM)                                  │
│     ├── LayerNorm (polynomial approx, rotation-free)          │
│     ├── FFN (PCMM × 2)                                       │
│     ├── GELU/SiLU (polynomial approx)                         │
│     └── [Bootstrapping as needed]                             │
│     │                                                         │
│     ▼                                                         │
│  Return ciphertext to client                                  │
└──────────────────────────────────────────────────────────────┘
```

### 6.1 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **HE Scheme** | RNS-CKKS | Standard for approximate arithmetic on real-valued hidden states |
| **Interaction model** | Non-interactive (pure FHE) | No MPC rounds → simpler protocol, lower communication |
| **Split point** | After embedding + 1 layer (client), bulk on server, last 1-2 layers + head on client | Minimizes ciphertext data sent; client handles sampling |
| **Nonlinear approx** | Polynomial (Chebyshev/minimax) for Softmax, LayerNorm, GELU | Required for non-interactive; follow MOAI/Euston approach |
| **Rotation strategy** | Column packing + zero-rotation nonlinear (MOAI/ARION) | Rotation is #1 bottleneck |
| **Attention variant** | Consider Gaussian kernel attention (replace softmax) | Eliminates hardest nonlinear function; per ICLR 2025 paper |
| **KV-cache** | Encrypted KV-cache reuse (CryptoGen approach) | Essential for autoregressive generation |
| **GPU library** | Phantom (MOAI's choice) or FIDESlib | GPU acceleration required for target throughput |
| **Model weights** | Plaintext on server | Enables fast PCMM; model privacy via access control (not crypto) |

### 6.2 Critical Path to 2-10 tok/s

1. **Minimize layers on ciphertext** — Each HE layer adds ~seconds. Running 28 layers on CKKS is the bottleneck.
   - Option A: Run fewer layers encrypted (e.g., 12 of 28), rest on client
   - Option B: Use smaller model (7B → 1-3B parameter)
   - Option C: Distill into encryption-friendly architecture

2. **Eliminate rotations** — Use MOAI/ARION zero-rotation techniques
   - Column packing for PCMM (rotation-free)
   - Polynomial Softmax/LayerNorm without rotation

3. **GPU acceleration** — Phantom/FIDESlib on A100/H200
   - MOAI achieves 22× speedup for Softmax on GPU
   - Bootstrapping is the remaining bottleneck

4. **Encrypted KV-cache** — Avoid quadratic re-computation
   - CryptoGen's approach: unified cache framework with heterogeneous SIMD encodings

5. **LoRA / adapter architecture** — Keep base weights plaintext (PCMM), only adapter weights encrypted if needed

---

## 7. Implementation Phases

### Phase 1: Prototype (CPU, Lattigo/TenSEAL)
- [ ] Set up CKKS encryption/decryption pipeline
- [ ] Implement single transformer layer on ciphertext (PCMM for linear, poly approx for nonlinear)
- [ ] Benchmark single-layer latency
- [ ] Client-server protocol (gRPC or similar)

### Phase 2: Multi-Layer + Optimization (CPU)
- [ ] Stack N layers with bootstrapping management
- [ ] Implement MOAI-style zero-rotation Softmax/LayerNorm
- [ ] Column packing for rotation-free PCMM
- [ ] Benchmark end-to-end on small model (BERT-tiny or GPT-2 small)

### Phase 3: GPU Acceleration (Phantom/FIDESlib)
- [ ] Port CKKS pipeline to GPU
- [ ] Implement optimized HE kernels for attention
- [ ] Benchmark on BERT-base / GPT-2 medium
- [ ] Target: approach 2 tok/s for generation

### Phase 4: Production Architecture
- [ ] Encrypted KV-cache (CryptoGen approach)
- [ ] Encryption-friendly model architecture (Gaussian kernel attention, LoRA)
- [ ] Scale to larger models (LLaMA-3-8B or similar)
- [ ] Target: 2-10 tok/s depending on model size and layer count

---

## 8. Key References

### Core Papers (Must-Read)
1. [MOAI — Zero-rotation secure transformer inference (NTU DTC)](https://eprint.iacr.org/2025/991)
2. [ARION — Double-BSGS attention optimization](https://eprint.iacr.org/2025/2271)
3. [Euston — S&P 2026, SVD-based efficient STFI](https://eprint.iacr.org/2026/046)
4. [BLB — Hybrid CKKS+MPC, USENIX Security 2025](https://eprint.iacr.org/2025/1532)
5. [CryptoGen — Encrypted KV-cache for generation](https://arxiv.org/abs/2602.08798)
6. [Encryption-Friendly LLM Architecture — ICLR 2025](https://arxiv.org/abs/2410.02486)

### NTU DTC Papers
7. [Efficient FHE-based PE-NN (NTU/HintSight, TDSC 2024)](https://eprint.iacr.org/2023/647)
8. [NTU Hybrid PP-NN (Split DNN)](https://www.ntu.edu.sg/innovates/tech-portal/tech-offers/detail/new-model-hybrid-privacy-preserving-neural-networks)
9. [PrivTuner — FHE + LoRA (NTU)](https://arxiv.org/abs/2410.00433)

### Surveys & SoKs
10. [Survey on Private Transformer Inference](https://arxiv.org/abs/2412.08145)
11. [SoK: Private DNN Inference with Approximate FHE (2026)](https://eprint.iacr.org/2026/047)

### GPU Libraries
12. [Phantom — CUDA HE library](https://github.com/encryptorion-lab/phantom-fhe)
13. [FIDESlib — Open-source CKKS on GPU](https://arxiv.org/abs/2507.04775)
14. [Lattigo — Go HE library](https://github.com/tuneinsight/lattigo)

### Foundational
15. [BOLT — IEEE S&P 2024](https://www.semanticscholar.org/paper/BOLT:-Privacy-Preserving,-Accurate-and-Efficient-Pang-Zhu/0f7bbe9837026560a934de8a74d233678bd55f57)
16. [BumbleBee — NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)
17. [THOR — ACM CCS 2025](https://eprint.iacr.org/2024/1881.pdf)
18. [Iron — NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/64e2449d74f84e5b1a5c96ba7b3d308e-Paper-Conference.pdf)
19. [Split HE — arXiv 2022](https://arxiv.org/abs/2202.13351)
20. [StriaNet — Zero-rotation NN architecture](https://arxiv.org/abs/2601.21287)
21. [Cachemir — FHE generative inference with KV cache](https://arxiv.org/abs/2602.11470)

---

## 9. Open Questions

1. **"N2HE" paper** — Could not locate a specific paper with this title from NTU DTC. The closest NTU DTC work is the Hybrid PP-NN / MOAI. Is this the correct reference, or is it a different acronym?

2. **Model choice** — What base model to target? (BERT-base for classification, GPT-2 for generation, LLaMA-3 for full LLM?)

3. **Threat model** — Pure data privacy (server sees encrypted input, model weights in plaintext)? Or also model privacy (weights encrypted)?

4. **Target hardware** — Consumer GPU (RTX 4090), datacenter GPU (A100/H200), or CPU-only prototype first?

5. **Interaction model** — Fully non-interactive (pure FHE, higher compute) vs. hybrid HE+MPC (lower compute, requires communication rounds)?
