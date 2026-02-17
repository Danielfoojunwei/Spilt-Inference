"""gRPC service definitions for the split inference protocol.

Extends TenSafe's HAS (Homomorphic Adapter Service) protocol with:
- ForwardSplit: Single round trip carrying DP states + encrypted deltas + GateLink signals
- NegotiateSplit: Client capability → server adapts split parameters
- UploadEncryptedAdapter: One-time AOE adapter upload

In the fused single round trip protocol:
- Client sends: DP-noised hidden states (one message)
- Server responds: h_base + all encrypted deltas + all GateLink signals (one message)
This collapses GateLink's 2×N_layers round trips into ONE.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.common.config import SplitInferenceConfig
from src.common.types import (
    NegotiateRequest,
    NegotiateResponse,
    SplitForwardRequest,
    SplitForwardResponse,
    UploadAdapterRequest,
)
from src.compiler.device_profiles import auto_detect_profile
from src.compiler.privacy_budget import PrivacyBudgetOptimizer
from src.server.parallel_helora import CKKSEngine, EncryptedAdapter
from src.server.split_server import SplitInferenceServer

logger = logging.getLogger(__name__)


class SplitInferenceServicer:
    """gRPC servicer for split inference protocol.

    In production, this wraps the SplitInferenceServer and exposes it
    via gRPC. For testing, it can be used directly.

    Implements three RPCs:
    1. ForwardSplit: The core inference RPC (one round trip per token)
    2. NegotiateSplit: Client announces capabilities, server returns optimal params
    3. UploadEncryptedAdapter: Client uploads pre-encrypted adapter (AOE)
    """

    def __init__(self, server: SplitInferenceServer, config: SplitInferenceConfig):
        self.server = server
        self.config = config
        self._optimizer = PrivacyBudgetOptimizer(
            total_model_layers=config.total_layers,
        )

    def ForwardSplit(self, request: SplitForwardRequest) -> SplitForwardResponse:
        """Process one split inference step.

        Single round trip: client sends DP-noised states, server returns
        base output + encrypted deltas + GateLink signals.
        """
        return self.server.forward_split(request)

    def NegotiateSplit(self, request: NegotiateRequest) -> NegotiateResponse:
        """Negotiate split parameters based on client device capabilities.

        Client announces RAM, TEE availability, preferred epsilon.
        Server returns optimized K, epsilon, rank, adapter_id.
        """
        profile = auto_detect_profile(
            available_ram_gb=request.available_ram_gb,
            has_tee=request.has_tee,
        )

        # Override with client preferences if specified
        epsilon = request.preferred_epsilon or profile.epsilon
        K = request.max_client_layers if request.max_client_layers is not None else profile.num_client_layers

        result = self._optimizer.optimize(profile)

        return NegotiateResponse(
            num_client_layers=K,
            epsilon=epsilon,
            lora_rank=result.lora_rank,
            adapter_id="default",
            total_model_layers=self.config.total_layers,
        )

    def UploadEncryptedAdapter(self, request: UploadAdapterRequest) -> str:
        """Upload pre-encrypted adapter (AOE: one-time upload).

        Client encrypts B matrices once and uploads them. Server stores
        the encrypted adapter and uses it for all subsequent requests.
        A matrices are sent in plaintext (FFA-LoRA: freeze-A).
        """
        adapter = EncryptedAdapter(
            adapter_id=request.adapter_id,
            encrypted_B=request.encrypted_B_matrices,
            plaintext_A=request.plaintext_A_matrices,
            gate_A=request.gate_A_matrices,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_alpha,
            num_layers=request.num_layers,
        )
        self.server.register_adapter(adapter)

        logger.info(
            "Adapter '%s' uploaded: rank=%d, layers=%d",
            request.adapter_id,
            request.lora_rank,
            request.num_layers,
        )
        return request.adapter_id
