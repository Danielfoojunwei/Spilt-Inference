"""Full split inference client: embed + DP noise + send + decrypt + sample.

Orchestrates the complete client-side pipeline for the DP-HE three-layer
privacy protocol:

1. INPUT PRIVACY: tokenize → embed → K layers → DP noise → send to server
2. ADAPTER PRIVACY: receive encrypted deltas → decrypt → GateLink gate eval
3. OUTPUT PRIVACY: combine h_base + gated deltas → LM head → sample locally

The server never sees raw tokens, adapter weights, or output logits.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

from src.client.decrypt import CKKSDecryptAssembler
from src.client.dp_noise import DPNoiseInjector
from src.client.model_shard import ClientModelShard
from src.common.config import SplitInferenceConfig
from src.common.types import SplitForwardRequest, SplitForwardResponse

logger = logging.getLogger(__name__)


class SplitInferenceClient:
    """End-to-end split inference client with three-layer privacy.

    Combines:
    - ClientModelShard: embedding + K layers + LM head (input/output privacy)
    - DPNoiseInjector: calibrated noise for hidden state privacy
    - CKKSDecryptAssembler: delta decryption + GateLink gate evaluation
    """

    def __init__(
        self,
        config: SplitInferenceConfig,
        server: Optional[object] = None,
        ckks_secret_key: Optional[bytes] = None,
    ):
        """Initialize split inference client.

        Args:
            config: Split inference configuration (from compiler).
            server: Server instance for local testing, or None for gRPC.
            ckks_secret_key: CKKS secret key for decryption. None = simulation.
        """
        self.config = config

        # Client model shard: embed + K layers + LM head
        self.shard = ClientModelShard(
            model_id=config.model_id,
            num_client_layers=config.num_client_layers,
            device="cpu",
        )

        # DP noise injector (Layer 1: input privacy)
        self.dp = DPNoiseInjector(
            epsilon=config.privacy.epsilon,
            delta=config.privacy.delta,
            sensitivity=config.privacy.sensitivity,
        )

        # CKKS decryptor + GateLink evaluator (Layer 2: adapter privacy)
        self.assembler = CKKSDecryptAssembler(ckks_secret_key=ckks_secret_key)

        # Server connection (local or gRPC)
        self._server = server  # Direct reference for local testing
        self._stub = None  # gRPC stub for production

    def load(self) -> None:
        """Load client model shard."""
        self.shard.load()
        logger.info(
            "Split client loaded: model=%s, K=%d, ε=%.1f",
            self.config.model_id,
            self.config.num_client_layers,
            self.config.privacy.epsilon,
        )

    def _send_to_server(self, request: SplitForwardRequest) -> SplitForwardResponse:
        """Send request to server (local or gRPC)."""
        if self._server is not None:
            # Local mode: direct function call
            return self._server.forward_split(request)
        elif self._stub is not None:
            # gRPC mode
            raise NotImplementedError("gRPC client not yet implemented")
        else:
            raise RuntimeError("No server configured (set server or gRPC stub)")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text with three-layer privacy.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text (including prompt).
        """
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature
        top_p_val = top_p or self.config.top_p

        # Tokenize (private: stays on client)
        input_ids = self.shard.tokenize(prompt)
        all_token_ids = input_ids.squeeze(0).tolist()

        start_time = time.perf_counter()
        tokens_generated = 0

        for step in range(max_tokens):
            step_start = time.perf_counter()

            # === LAYER 1: INPUT PRIVACY ===
            # Embed + K client layers (private, local)
            current_ids = torch.tensor([all_token_ids], dtype=torch.long)
            h_K = self.shard.embed_and_forward(current_ids)

            # DP noise injection (ε-differential privacy)
            h_noised, noise_stats = self.dp.inject_noise(h_K)

            # === SEND TO SERVER (one request) ===
            request = SplitForwardRequest(
                hidden_states=h_noised.squeeze(0).numpy(),
                token_positions=list(range(len(all_token_ids))),
                adapter_id="default",
                sequence_id="gen-0",
            )
            response = self._send_to_server(request)

            # === LAYER 2: ADAPTER PRIVACY ===
            # Decrypt deltas + evaluate GateLink gates (private, local)
            h_final = self.assembler.assemble(response)

            # === LAYER 3: OUTPUT PRIVACY ===
            # LM head + sampling (private, local — logits never leave client)
            next_token_id = self.shard.decode_and_sample(
                h_final, temperature=temp, top_p=top_p_val
            )

            all_token_ids.append(next_token_id)
            tokens_generated += 1

            step_ms = (time.perf_counter() - step_start) * 1e3
            logger.debug(
                "Step %d: token=%d, latency=%.1f ms, SNR=%.1f dB",
                step,
                next_token_id,
                step_ms,
                noise_stats.snr_db,
            )

            # Check for EOS
            if next_token_id == self.shard.eos_token_id:
                break

        total_time = time.perf_counter() - start_time
        throughput = tokens_generated / total_time if total_time > 0 else 0

        logger.info(
            "Generation complete: %d tokens in %.2f s (%.1f tok/s), ε=%.1f",
            tokens_generated,
            total_time,
            throughput,
            self.config.privacy.epsilon,
        )

        return self.shard.detokenize(all_token_ids)
