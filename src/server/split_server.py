"""Split inference server: base model forward + parallel HE-LoRA.

Orchestrates the server side of the DP-HE three-layer split protocol:
1. Receives DP-noised hidden states from client
2. Runs base model layers [K, N) in plaintext (full GPU speed)
3. Feeds each layer's hidden states to the parallel HE-LoRA executor
4. Returns h_base + all encrypted deltas + all GateLink signals in ONE response

The base model and HE-LoRA paths are decoupled (parallel adapter injection):
- Base path: sequential transformer layers, plaintext, no adapter feedback
- HE-LoRA path: parallel computation of encrypted deltas per layer
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.common.config import SplitInferenceConfig
from src.common.types import SplitForwardRequest, SplitForwardResponse
from src.server.parallel_helora import CKKSEngine, EncryptedAdapter, ParallelHELoRAExecutor

logger = logging.getLogger(__name__)


class SplitInferenceServer:
    """Server-side split inference engine.

    Holds the base model layers [K, N) and the parallel HE-LoRA executor.
    Processes requests in a single forward pass + parallel HE computation,
    returning everything in one fused response.
    """

    def __init__(self, config: SplitInferenceConfig):
        self.config = config
        self._server_layers: Optional[nn.ModuleList] = None
        self._hidden_size: int = 0
        self._helora: Optional[ParallelHELoRAExecutor] = None
        self._loaded = False

    def load(self) -> None:
        """Load server-side model components and initialize HE engine."""
        from transformers import AutoConfig, AutoModelForCausalLM

        logger.info(
            "Loading server model: %s, layers [%d, %d)",
            self.config.model_id,
            self.config.num_client_layers,
            self.config.num_client_layers + self.config.num_server_layers,
        )

        config = AutoConfig.from_pretrained(self.config.model_id)
        self._hidden_size = config.hidden_size

        K = self.config.num_client_layers
        N = K + self.config.num_server_layers

        # Load full model and extract server layers
        full_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float32,
        )

        self._server_layers = nn.ModuleList(
            [full_model.model.layers[i] for i in range(K, N)]
        )
        for layer in self._server_layers:
            layer.requires_grad_(False)

        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize CKKS engine and HE-LoRA executor
        ckks = CKKSEngine(
            poly_modulus_degree=self.config.privacy.poly_modulus_degree,
            coeff_mod_bit_sizes=self.config.privacy.coeff_mod_bit_sizes,
            scale_bits=self.config.privacy.scale_bits,
        )
        self._helora = ParallelHELoRAExecutor(
            ckks_engine=ckks,
            gatelink_enabled=self.config.privacy.gatelink_enabled,
        )

        self._loaded = True
        logger.info("Server loaded: %d layers, HE-LoRA ready", N - K)

    def register_adapter(self, adapter: EncryptedAdapter) -> None:
        """Register a pre-encrypted adapter (AOE: uploaded once by client)."""
        assert self._helora is not None, "Call load() first"
        self._helora.register_adapter(adapter)

    @torch.no_grad()
    def forward_split(self, request: SplitForwardRequest) -> SplitForwardResponse:
        """Process one split inference step.

        Runs the base model forward pass and parallel HE-LoRA computation,
        returning everything in one fused response.

        Args:
            request: Contains DP-noised hidden states from client.

        Returns:
            Fused response with base output + encrypted deltas + GateLink signals.
        """
        assert self._loaded, "Call load() first"
        start_time = time.perf_counter()

        # Convert input to tensor
        h = torch.from_numpy(request.hidden_states).float()
        if h.dim() == 2:
            h = h.unsqueeze(0)  # Add batch dimension

        # === BASE PATH (plaintext, full speed) ===
        K = self.config.num_client_layers
        layer_hidden_states: dict[int, np.ndarray] = {}

        for i, layer in enumerate(self._server_layers):
            layer_idx = K + i

            # Store hidden states for HE-LoRA path (parallel adapter)
            layer_hidden_states[layer_idx] = h.squeeze(0).numpy().copy()

            # Forward through base transformer layer
            layer_output = layer(h)
            if isinstance(layer_output, tuple):
                h = layer_output[0]
            else:
                h = layer_output

        h_base = h.squeeze(0).numpy()

        base_time_ms = (time.perf_counter() - start_time) * 1e3

        # === HE-LORA PATH (parallel, encrypted) ===
        he_start = time.perf_counter()
        encrypted_deltas = []
        gatelink_signals = []

        if self._helora and request.adapter_id in self._helora._adapters:
            encrypted_deltas, gatelink_signals = self._helora.compute_all_deltas(
                adapter_id=request.adapter_id,
                layer_hidden_states=layer_hidden_states,
            )

        he_time_ms = (time.perf_counter() - he_start) * 1e3
        total_ms = (time.perf_counter() - start_time) * 1e3

        logger.info(
            "Forward split: base=%.1f ms, HE-LoRA=%.1f ms, total=%.1f ms, layers=%d",
            base_time_ms,
            he_time_ms,
            total_ms,
            len(self._server_layers),
        )

        return SplitForwardResponse(
            base_hidden_states=h_base,
            encrypted_deltas=encrypted_deltas,
            gatelink_signals=gatelink_signals,
            layers_computed=len(self._server_layers),
        )
