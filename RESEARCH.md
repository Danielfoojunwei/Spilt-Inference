# HE Split Inference — Research, Novelty & Architecture Plan

## Building on TenSafe

This work extends [TenSafe](https://github.com/Danielfoojunwei/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation) (v4.1.0), which provides three foundational innovations:

### TenSafe Innovation 1: ZeRo-MOAI (Paper 1)
**Column-packing that eliminates ALL rotation operations from encrypted LoRA.**
- Rotations account for >93% of CKKS computation time. LoRA's rank-deficiency (r << d) allows column-packing that avoids rotations entirely.
- **14.9x speedup** over naive HE-LoRA. Zero Galois keys (eliminates 2.4 GB → 0 MB key material).
- Rank-independent cost: increasing LoRA rank improves quality with zero additional HE overhead.
- Setup time: ~25s → <100ms (250x improvement).

### TenSafe Innovation 2: Speculative Batching (Paper 2)
**Pack K draft tokens into SIMD slots for parallel encrypted verification.**
- Sequential token generation wastes 99.9% of SIMD capacity (32/4096 slots used).
- Uses the plaintext base model as a draft model (>95% accurate predictor of adapted output).
- **3.8x throughput improvement** (56.8x combined with ZeRo-MOAI).
- K scales to 128 tokens before slot saturation.

### TenSafe Innovation 3: GateLink Protocol (Paper 3)
**Client-aided non-linear evaluation — zero approximation error.**
- Problem: Polynomial approximations of SiLU accumulate "cumulative error >= 226 over 36 layers" and cost 2,977 ms/layer.
- GateLink: Server sends encrypted rank-r pre-activation to client → client decrypts and evaluates exact non-linear function → returns gate bit → server applies gated LoRA.
- **5.4 ms datacenter round-trip**, 2.2x faster than degree-3 polynomial approximation.
- Enables gated LoRA: `y = Wx + g(x) * B(Ax)` for conditional expert activation / MoE routing.
- Eliminates ALL special keys when combined with ZeRo-MOAI.

### TenSafe Baseline Performance (A100, rank r=32)

| Architecture | Llama 8B | HE Overhead | Kimi 2.5 (MoE) | HE Overhead |
|:---|:---|:---|:---|:---|
| Standard vLLM (FP16) | 53.18 tok/s | 1.0x | 25.00 tok/s | 1.0x |
| TenSafe (A100) | 5.76 tok/s | 9.2x | 3.37 tok/s | 7.4x |
| TenSafe (Groq LPU) | 28.78 tok/s | 1.8x | 7.71 tok/s | 3.2x |
| Vanilla HE-LoRA | 2.22 tok/s | 24.0x | 0.50 tok/s | 50.0x |
| Full HE LLM | 0.05 tok/s | 1000x+ | DNF | N/A |

---

## 1. The Critical Gap: Hidden State Privacy

### 1.1 The Problem TenSafe Doesn't Solve

TenSafe protects **adapter weights** (CKKS encryption) and uses **TEE** (hardware trust) for input/base model protection. But:

1. **TEE limits deployment** — Requires Intel SGX/TDX. Most consumer devices (phones, ARM laptops, AMD workstations) lack TEE. TenSafe's input privacy relies on hardware that most users don't have.

2. **Hidden states leak input text** — Research proves that intermediate hidden states can be inverted to reconstruct input tokens:
   - **Prompt Inversion Attack (CCS 2025)**: >90% reconstruction accuracy across Llama-3.2, Phi-3.5, GPT-2, BERT.
   - **Constrained Optimization Attack (2025)**: 88.4% token accuracy on Llama-65B from maximum layer inversion.
   - **Embedding Inversion (Morris et al. 2023)**: 92% recovery of 32-token input from T5 embeddings.
   - **Diffusion-based attacks (2024)**: 81.3% token recovery without access to target encoder.

3. **Sending raw hidden states to server = no input privacy.** An adversary with the public embedding matrix can near-perfectly reconstruct input text.

### 1.2 What's Needed

A mechanism that protects input privacy **without TEE** and **without full FHE** (which is 1000x too slow). The mechanism must:
- Work on commodity hardware (phones, laptops)
- Add minimal latency (hidden behind base model compute)
- Have formal mathematical privacy guarantees
- Compose with TenSafe's existing HE-LoRA pipeline

**Solution: Differential Privacy (DP) noise on hidden states**, validated by:
- **Split-and-Denoise (SnD, ICML 2024)**: Correlation drops below 0.005 with calibrated noise.
- **DEL (2025)**: Projection + DP quantization overcomes curse of dimensionality.
- **NVDP (2025)**: Variational information bottleneck with Rényi DP guarantees.

---

## 2. Novel Architecture: DP-HE Three-Layer Split

### 2.1 The Core Insight

**Use the RIGHT tool for each privacy threat:**

| Threat | Protection | Mechanism | Guarantee | Overhead |
|--------|-----------|-----------|-----------|----------|
| Server reconstructs input tokens | DP noise on hidden states | Calibrated Gaussian noise before transmission | ε-differential privacy | ~0 (noise addition is free) |
| Server learns adapter weights | CKKS HE via ZeRo-MOAI | Encrypted B matrices, zero-rotation PCMM | IND-CPA under RLWE | ~812 μs/layer |
| Server learns output distribution | Client-side LM head + sampling | Logits never transmitted | Information-theoretic | ~1-5 ms |
| Non-linear approx error | GateLink protocol | Client-side exact evaluation | Zero error | Piggybacked on split round trip |

**No existing system combines all four.** Full FHE systems (MOAI, ARION, Euston) encrypt everything at 1000x overhead. Split learning systems (SnD, DEL) protect input but not adapters. HE-LoRA systems (CryptPEFT, PrivTuner) protect adapters but not input. We provide **three-layer privacy at near-plaintext speed.**

### 2.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT                                  │
│  (phone/laptop/workstation — 1-2 GB RAM)                         │
│                                                                  │
│  tokens = tokenize(input_text)              # Private, local     │
│  h_0 = embedding(tokens)                    # Private, local     │
│  h_K = transformer_layers[0:K](h_0)         # K=1-2 layers       │
│  noise = dp_gaussian(ε, δ, sensitivity)     # DP mechanism        │
│  h_sent = h_K + noise                       # ε-DP protected     │
│         │                                                        │
│         ▼  ──── ONE request per token ────────────────►          │
│                                                                  │
│  h_base       ◄── base model output (plaintext)                  │
│  enc_deltas[] ◄── encrypted LoRA deltas (CKKS ciphertexts)       │
│  pre_gates[]  ◄── encrypted pre-activations (GateLink, rank-r)   │
│                                                                  │
│  For each layer i:                                               │
│    gate_i = exact_nonlinear(decrypt(pre_gates[i]))  # GateLink   │
│    delta_i = decrypt(enc_deltas[i])                  # CKKS       │
│    gated_delta_i = gate_i * delta_i                  # Gated LoRA │
│                                                                  │
│  h_final = h_base + Σ gated_delta_i                              │
│  logits = lm_head(h_final)                  # Private, local     │
│  next_token = sample(logits, top_p)         # Private, local     │
│         │                                                        │
│         ▼  loop for autoregressive generation                    │
└─────────────────────────────────────────────────────────────────┘
                     │                    ▲
                     ▼                    │
┌─────────────────────────────────────────────────────────────────┐
│                          SERVER                                  │
│  (GPU — A100/H200/4090)                                          │
│                                                                  │
│  h_sent ◄── receive DP-noised hidden states                      │
│                                                                  │
│  ┌─── BASE PATH (plaintext, full GPU speed) ─────────────┐      │
│  │  For each layer i in K..N:                             │      │
│  │    h_i = transformer_layer_i(h_{i-1})   # Normal fwd  │      │
│  │  h_base = h_N                                         │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌─── HE-LORA PATH (parallel, encrypted) ────────────────┐      │
│  │  For each layer i in K..N:                             │      │
│  │    x_i = h_i from base path                            │      │
│  │    z_i = encrypted_B_i @ x_i     # PCMM (ZeRo-MOAI)  │      │
│  │    delta_i = A_i @ z_i           # PCMM (A plaintext) │      │
│  │    pre_gate_i = A_gate_i @ z_i   # GateLink signal    │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  Send: h_base + [enc_delta_i] + [pre_gate_i]  ────────►         │
│                                                                  │
│  Server NEVER sees:                                              │
│    ✗ Raw tokens (client embeds locally)                           │
│    ✗ Clean hidden states (only DP-noised version)                 │
│    ✗ Adapter weights (encrypted under CKKS)                       │
│    ✗ Gate decisions (client evaluates locally)                     │
│    ✗ Output logits or sampled tokens (client-side)                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Protocol Property: Single Round Trip

**Standard GateLink** requires 2 × N_layers round trips per token:
- Per layer: server sends pre-activation → client returns gate bit → server applies

**Our fused protocol** collapses this to **ONE round trip per token**:
1. Client → Server: DP-noised `h_sent` (one transmission)
2. Server: runs base model + computes ALL encrypted LoRA data (parallel path)
3. Server → Client: `h_base` + all `enc_delta_i` + all `pre_gate_i` (one response)
4. Client: decrypts everything locally, evaluates gates, combines, samples

This works because the **parallel adapter path** decouples LoRA from the base forward pass. All layers' pre-activations and deltas are available after one base model forward pass — no sequential gate-bit exchange needed.

**Communication reduction: 2 × N_layers → 1 round trip per token** (e.g., 64 → 1 for a 32-layer model).

---

## 3. Novel Contributions (Paper-Ready)

### Contribution 1: DP-HE Three-Layer Privacy Architecture

**First system combining DP (input) + HE (adapter) + local compute (output).**

Full FHE encrypts everything → impractical (1000x overhead). Split learning adds DP noise → protects input but not adapters. HE-LoRA encrypts adapters → protects adapters but not input. We compose all three at <10x overhead.

**Formal guarantees:**
- Input: ε-differential privacy (tunable ε per device)
- Adapter: IND-CPA security under Ring-LWE
- Output: information-theoretic (never transmitted)

**Why novel:** No prior system achieves all three simultaneously. CryptPEFT (2024) is closest — confines privacy to adapters but provides NO input protection. SnD (ICML 2024) protects input but provides NO adapter protection.

### Contribution 2: GateLink-Fused Single-Round-Trip Protocol

**Collapse GateLink's multi-round gate exchange into one request-response.**

Standard GateLink: 2N round trips per token (send pre-activation, receive gate bit, per layer).
Our protocol: 1 round trip per token (send DP states, receive everything).

This is possible because:
1. **Parallel adapter path** — base model runs all layers first, then all HE-LoRA data is available
2. **Client-side gate evaluation** — client already has decryption key, so it evaluates gates locally
3. **GateLink data piggybacks** — pre-activations and deltas ride the same response as h_base

**Latency improvement:** For 32-layer model on datacenter network (5.4ms RTT per GateLink exchange): Standard GateLink = 32 × 5.4ms = 172.8ms. Our protocol = 1 × ~10ms = 10ms. **17x reduction in protocol overhead.**

### Contribution 3: Adapter-Only Encryption with ZeRo-MOAI

**Encrypt adapter weights once at upload, not activations per-token.**

Standard HE-LoRA: encrypt(activations) → PCMM with plaintext weights → decrypt(delta).
AOE: plaintext activations → PCMM with encrypt(adapter weights) → encrypted delta → client decrypts.

Both are PCMM — same cost under ZeRo-MOAI's column packing. But AOE eliminates:
- Per-token client-side encryption (costly on phones)
- Key management per session (adapter encrypted once, stored on server)
- Bandwidth for per-token ciphertext upload (adapters uploaded once)

Combined with FFA-LoRA (freeze-A): only B matrices are encrypted. A is public/plaintext. Per-token cost = N_layers × (1 PCMM + 1 decrypt). Client needs only CKKS decryption — much lighter than encryption.

### Contribution 4: DP-Aware Speculative Batching

**Speculative batching works with DP-noised input because the client can draft on clean data.**

The key insight: in our split, the client has the CLEAN hidden states h_K (before noise). The client can:
1. Run a lightweight draft model (or the same K client layers + a small head) on clean h_K
2. Draft K candidate tokens
3. Pack K tokens' DP-noised states into one SIMD ciphertext
4. Server verifies all K in one encrypted forward pass

This preserves TenSafe's 3.8x speculative batching speedup while adding DP protection. The drafting happens on clean data (client-side), so draft quality isn't degraded by DP noise. Only the verification happens on noisy data, where the LoRA delta corrections still match because they're computed on the same noisy states the base model saw.

### Contribution 5: Device-Adaptive Privacy Compilation

**Joint optimization of (ε, K, rank, block_size) per device capability.**

| Profile | K (client layers) | ε (DP budget) | LoRA rank | Client RAM | Expected overhead |
|---------|-------------------|---------------|-----------|------------|-------------------|
| Phone | 1 | 1.0 | 4 | 1.5 GB | ~8x |
| Laptop | 1 | 4.0 | 8 | 3.0 GB | ~5x |
| Workstation | 2 | 8.0 | 16 | 6.0 GB | ~3x |
| Server (no TEE) | 4 | 16.0 | 32 | 16.0 GB | ~2x |
| Server (with TEE) | 0 | ∞ (TEE) | 32 | 0 (TEE) | ~1.5x |

The compiler generates device-specific split schedules. Lower-capability devices get more DP noise (stronger privacy, more quality loss) and lower LoRA rank (less HE work). This is a continuous trade-off, not a binary choice.

### Contribution 6: Parallel Encrypted Adapter Injection

**Decoupled base path and HE-LoRA path for pipeline parallelism.**

Research validates minimal quality loss from parallel adapter injection:
- **Side-Tuning (ECCV 2020)**: Best average rank across benchmarks with fully decoupled adapter.
- **MAM Adapters (ICLR 2022)**: Parallel config = "best among adapter/prompt methods."
- **LLaMA-Adapter (ICLR 2024)**: Zero-init parallel injection matches full fine-tuning.

In our system: base forward pass and HE-LoRA computation run as independent streams on the GPU. The base path feeds hidden states to the HE-LoRA path (one-way dependency), but HE-LoRA output doesn't feed back into the base path. This enables:
- True pipeline overlap (base layer N+1 overlaps with HE-LoRA layer N)
- Single-pass base model computation (no wait for HE results between layers)
- All HE data available at end of forward pass (enabling single round trip)

---

## 4. Composing TenSafe Innovations in Split Inference

### 4.1 How Each TenSafe Innovation Maps to Split Inference

| TenSafe Innovation | Role in Split Inference | What Split Inference Adds |
|---|---|---|
| **ZeRo-MOAI** | Server-side encrypted LoRA computation (zero rotations, zero keys) | AOE mode: encrypt weights not activations. Same PCMM, flipped operands |
| **Speculative Batching** | Pack K draft tokens for parallel encrypted verification | Client-side drafting on clean data (before DP noise) preserves draft quality |
| **GateLink** | Non-linear gate evaluation for gated LoRA | Fused into single round trip — client evaluates ALL gates locally, no per-layer exchange |
| **TEE base model** | Hardware trust for input + base model protection | **REPLACED by DP noise** for commodity hardware. TEE remains optional for maximum speed |
| **FFA-LoRA (Freeze-A)** | 50% communication reduction (only B is private) | Combined with AOE: only B encrypted at upload, A is always plaintext |
| **HAS gRPC service** | Client-server communication protocol | Extended with `ForwardSplit` RPC, `NegotiateSplit`, `UploadEncryptedAdapter` |
| **Cost-budgeted compiler** | ≤16 rotations/token, ≤64/layer enforcement | Extended with privacy budget parameters (ε, K, device profile) |

### 4.2 The Full Innovation Stack

```
┌────────────────────────────────────────────────────┐
│  Device-Adaptive Compiler (Contribution 5)          │  ← NEW: Joint (ε,K,rank) optimization
│  Extends TenSafe cost-budgeted compiler             │
├────────────────────────────────────────────────────┤
│  DP-HE Three-Layer Protocol (Contribution 1)        │  ← NEW: DP + HE + local compute
│  Fused Single Round Trip (Contribution 2)           │  ← NEW: Collapses GateLink exchanges
├────────────────────────────────────────────────────┤
│  Parallel Adapter Injection (Contribution 6)        │  ← NEW: Decoupled base + HE-LoRA paths
│  DP-Aware Speculative Batching (Contribution 4)     │  ← NEW: Client-side clean drafting
├────────────────────────────────────────────────────┤
│  GateLink Protocol                                  │  ← FROM TENSAFE: Non-linear gates
│  ZeRo-MOAI PCMM                                    │  ← FROM TENSAFE: Zero-rotation engine
│  Speculative Batching                               │  ← FROM TENSAFE: SIMD utilization
│  FFA-LoRA / AOE (Contribution 3)                    │  ← EXTENDED: Adapter-only encryption
├────────────────────────────────────────────────────┤
│  CKKS Crypto Backend (Pyfhel / N2HE)               │  ← FROM TENSAFE: HE primitives
│  HAS gRPC Service                                   │  ← FROM TENSAFE: Communication layer
│  vLLM Adapter + Hook Manager                        │  ← FROM TENSAFE: Base model serving
└────────────────────────────────────────────────────┘
```

---

## 5. Why This Matters: The Deployment Story

### 5.1 TenSafe's Current Limitation

TenSafe achieves 5.76 tok/s on A100 — impressive for HE, but requires:
- Server with TEE (Intel SGX/TDX) for input privacy
- Client capable of CKKS encryption per token
- Datacenter-class networking for GateLink round trips

This limits deployment to: **enterprise datacenter ↔ enterprise client.**

### 5.2 What Split Inference Enables

**Any device, any network, practical privacy:**

| Scenario | Before (TenSafe) | After (Split Inference) |
|----------|------------------|------------------------|
| Phone user | Impossible (no TEE, no encryption capability) | Works: embed + 1 layer + DP noise + decrypt only |
| Consumer laptop | Marginal (no TEE on most laptops) | Works: DP noise replaces TEE requirement |
| Edge/IoT | Impossible | Works (Tier 1): adapter-only encryption, no client-side layers |
| Enterprise datacenter | Works (current TenSafe) | Faster: parallel adapter + single round trip |
| Air-gapped / high-security | Requires TEE procurement | Works: DP provides mathematical guarantee, no hardware dependency |

### 5.3 Expected Performance (Projected)

| Configuration | Throughput | Privacy Level | Target Device |
|---|---|---|---|
| Split + DP(ε=1) + AOE + r=4 | ~8-15 tok/s | Strong (ε=1 DP + HE adapter) | Phone |
| Split + DP(ε=4) + AOE + r=8 | ~15-30 tok/s | Moderate (ε=4 DP + HE adapter) | Laptop |
| Split + DP(ε=8) + parallel + r=16 | ~30-45 tok/s | Light DP + HE adapter | Workstation |
| Split + TEE + parallel + r=32 | ~40-50 tok/s | Full (TEE + HE adapter) | Server |
| Vanilla vLLM (no privacy) | 53.18 tok/s | None | Reference |

---

## 6. SOTA Landscape

### 6.1 Pure FHE — What We're NOT Doing (and Why)

| System | Venue | Latency | Why We Diverge |
|--------|-------|---------|----------------|
| [NEXUS](https://eprint.iacr.org/2024/136.pdf) | NDSS 2025 | ~1103s (CPU) | Encrypts entire model |
| [MOAI](https://eprint.iacr.org/2025/991) | ePrint 2025 | Best GPU pure FHE | We use MOAI's zero-rotation for LoRA only |
| [ARION](https://eprint.iacr.org/2025/2271) | ePrint 2025 | 2.5x faster than MOAI | Double-BSGS useful for future encrypted attention |
| [Euston](https://eprint.iacr.org/2026/046) | S&P 2026 | SOTA pure FHE | SVD decomposition applicable to adapter encoding |
| [STIP](https://eprint.iacr.org/2026/174) | ePrint 2026 | 1.6x faster than Euston | Compact packing ideas reusable |

### 6.2 Split Learning + Privacy

| System | Venue | What it Does | Gap We Fill |
|--------|-------|-------------|-------------|
| [Split-and-Denoise (SnD)](https://arxiv.org/abs/2310.09130) | ICML 2024 | DP noise on embeddings + denoising | No adapter protection |
| [DEL](https://arxiv.org/html/2602.11513) | arXiv 2025 | Projection + DP quantization | No adapter protection |
| [PrivDFS](https://arxiv.org/html/2508.04346v1) | arXiv 2025 | Feature partitioning across servers | Requires non-colluding servers |
| [Eguard](https://arxiv.org/abs/2411.05034) | arXiv 2024 | Embedding projection + MI optimization | No adapter protection |

### 6.3 HE-LoRA / Adapter Privacy

| System | Venue | What it Does | Gap We Fill |
|--------|-------|-------------|-------------|
| [CryptPEFT](https://arxiv.org/abs/2412.08145) | arXiv 2024 | Confine privacy to adapters only | No input protection |
| [PrivTuner](https://arxiv.org/abs/2410.00433) | arXiv 2024 | FHE + LoRA fine-tuning | Training only, not inference. No input protection |
| TenSafe v4.1.0 | — | HE-LoRA + GateLink + speculative batching | Requires TEE for input protection |

### 6.4 Parallel Adapters (Quality Validation)

| System | Venue | Finding |
|--------|-------|---------|
| [Side-Tuning](https://arxiv.org/abs/1912.13503) | ECCV 2020 | Decoupled adapter achieves best avg rank across benchmarks |
| [MAM Adapters](https://openreview.net/forum?id=0RDcd5Axok) | ICLR 2022 | Parallel adapter = "best among adapter/prompt methods" |
| [LLaMA-Adapter](https://arxiv.org/abs/2303.16199) | ICLR 2024 | Zero-init parallel injection matches full fine-tuning |
| [Symbiosis](https://arxiv.org/html/2507.03220v1) | arXiv 2025 | Systems-level decoupling for multi-adapter serving |

### 6.5 Hidden State Attacks (Threat Validation)

| Attack | Venue | Result |
|--------|-------|--------|
| [Prompt Inversion](https://dl.acm.org/doi/pdf/10.1145/3719027.3744820) | CCS 2025 | >90% token reconstruction from hidden states |
| [Constrained Optimization](https://arxiv.org/html/2503.09022v1) | arXiv 2025 | 88.4% token accuracy on Llama-65B |
| [Embedding Inversion](https://arxiv.org/html/2405.11916) | arXiv 2024 | 92% recovery from T5 embeddings |
| [Model Inversion in Split Learning](https://arxiv.org/html/2501.05965) | arXiv 2025 | Information bottleneck analysis of split point privacy |

---

## 7. TenSafe Components We Build On

### 7.1 Already Implemented (in TenSafe)

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
| GateLink gate evaluator | `client/gate_evaluator.py` | Working |
| HAS gRPC server | `services/has/` | Working |
| MSS REST API | `services/mss/` | Working |
| vLLM adapter | `backend/vllm_adapter/` | Working (hook-based delta injection) |
| TenSafe Python client | `client/client.py` | Working |
| TypeScript SDK | `sdk/typescript/` | Working |
| N2HE CUDA backend | `he_lora_microkernel/n2he/` | Stub (architecture defined) |

### 7.2 What Split Inference Adds

| New Component | Purpose | Builds On |
|---------------|---------|-----------|
| **Client model shard** | Embed + K layers + LM head (lightweight client runtime) | HuggingFace model loading |
| **DP noise injector** | Calibrated Gaussian noise for hidden state privacy | New (SnD-inspired) |
| **GateLink-fused split protocol** | Single round trip carrying DP states + deltas + gates | GateLink + HAS extension |
| **Parallel HE-LoRA executor** | Decoupled adapter path for pipeline parallelism | TenSafe executor extension |
| **Adapter-Only Encryption mode** | Encrypt adapter weights at upload, not per-token | Compiler + PCMM mode flip |
| **Device-adaptive compiler** | Joint (ε, K, rank) optimization per device | Cost model extension |
| **DP-aware speculative batcher** | Client-side clean drafting + SIMD packing | Speculative batching extension |
| **Split negotiation RPC** | Client capability → server adapts protocol | HAS protocol extension |

---

## 8. Implementation Plan

### Phase 1: Project Setup & Client Model Shard
Extract embedding + K transformer layers + LM head from HuggingFace model as lightweight client runtime (~1-2 GB for Llama-3-8B K=1).

### Phase 2: DP Noise Injection
Calibrated Gaussian mechanism for hidden state privacy. ε-DP guarantees with auto-calibrated sensitivity.

### Phase 3: GateLink-Fused Split Protocol
Extend HAS gRPC with `ForwardSplit` RPC. Server returns h_base + encrypted deltas + GateLink pre-activations in one response. Client evaluates all gates locally.

### Phase 4: Parallel HE-LoRA Executor
Decoupled base path and adapter path. Pipeline overlap. Uses TenSafe's ZeRo-MOAI PCMM engine.

### Phase 5: Device-Adaptive Compiler
Joint optimization of privacy and performance parameters per device profile. Extends TenSafe cost model.

### Phase 6: End-to-End Integration & Benchmarks
Wire everything, validate privacy guarantees (prompt inversion attack on DP-noised states), measure throughput.

---

## 9. Key References

### Our Direct Lineage
1. [TenSafe — HE-LoRA Platform](https://github.com/Danielfoojunwei/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation)
2. [MOAI — Zero-rotation (NTU DTC)](https://eprint.iacr.org/2025/991)
3. [Hybrid PP-NN (NTU/HintSight)](https://eprint.iacr.org/2023/647)
4. [PrivTuner — FHE + LoRA (NTU)](https://arxiv.org/abs/2410.00433)

### Techniques We Compose
5. [Split-and-Denoise — DP for split learning (ICML 2024)](https://arxiv.org/abs/2310.09130)
6. [Side-Tuning — Parallel adapter validation (ECCV 2020)](https://arxiv.org/abs/1912.13503)
7. [MAM Adapters — Parallel adapter quality (ICLR 2022)](https://openreview.net/forum?id=0RDcd5Axok)
8. [LLaMA-Adapter — Zero-init parallel injection (ICLR 2024)](https://arxiv.org/abs/2303.16199)
9. [CryptPEFT — Adapter-only privacy (arXiv 2024)](https://arxiv.org/abs/2412.08145)
10. [Encryption-Friendly LLM (ICLR 2025)](https://arxiv.org/abs/2410.02486)

### The Competition
11. [NEXUS (NDSS 2025)](https://eprint.iacr.org/2024/136.pdf)
12. [THOR (CCS 2025)](https://eprint.iacr.org/2024/1881.pdf)
13. [BOLT (S&P 2024)](https://www.semanticscholar.org/paper/BOLT:-Privacy-Preserving,-Accurate-and-Efficient-Pang-Zhu/0f7bbe9837026560a934de8a74d233678bd55f57)
14. [BumbleBee (NDSS 2025)](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)
15. [Euston (S&P 2026)](https://eprint.iacr.org/2026/046)
16. [STIP (ePrint 2026)](https://eprint.iacr.org/2026/174)

### Privacy Attacks (Threat Model Validation)
17. [Prompt Inversion Attack (CCS 2025)](https://dl.acm.org/doi/pdf/10.1145/3719027.3744820)
18. [Constrained Optimization Inversion (arXiv 2025)](https://arxiv.org/html/2503.09022v1)
19. [Embedding Inversion (arXiv 2024)](https://arxiv.org/html/2405.11916)
20. [Model Inversion in Split Learning (arXiv 2025)](https://arxiv.org/html/2501.05965)

### Surveys
21. [SoK: Private DNN Inference with FHE (2026)](https://eprint.iacr.org/2026/047)
22. [Survey on Private Transformer Inference](https://arxiv.org/abs/2412.08145)
