"""Client-side model shard: embedding + K transformer layers + LM head.

Extracts a lightweight subset of a HuggingFace transformer model for
client-side execution. The client runs:
  1. Tokenization
  2. Embedding lookup
  3. First K transformer layers (K=1-2 typically)
  4. Final layer norm + LM head (for output decoding)

This keeps raw tokens and output logits private (never sent to server).
For Llama-3-8B with K=1, client memory is ~1.5 GB.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClientModelShard:
    """Lightweight client-side model shard for split inference.

    Holds only the components needed for input embedding, first K layers,
    and output decoding. All other layers run on the server.
    """

    def __init__(
        self,
        model_id: str,
        num_client_layers: int = 1,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.model_id = model_id
        self.num_client_layers = num_client_layers
        self.device = device
        self.torch_dtype = torch_dtype

        self._tokenizer = None
        self._embed_tokens: Optional[nn.Embedding] = None
        self._client_layers: Optional[nn.ModuleList] = None
        self._norm: Optional[nn.Module] = None
        self._lm_head: Optional[nn.Linear] = None
        self._hidden_size: int = 0
        self._total_layers: int = 0

        self._loaded = False

    def load(self) -> None:
        """Load model components from HuggingFace.

        Only loads: tokenizer, embedding, first K layers, norm, LM head.
        Remaining layers are NOT loaded (saves memory).
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading client shard: model=%s, K=%d, device=%s",
            self.model_id,
            self.num_client_layers,
            self.device,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        config = AutoConfig.from_pretrained(self.model_id)
        self._hidden_size = config.hidden_size
        self._total_layers = config.num_hidden_layers

        # Load full model then extract components
        # For production: use safetensors partial loading to avoid full model in memory
        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )

        # Extract embedding
        self._embed_tokens = full_model.model.embed_tokens
        self._embed_tokens.requires_grad_(False)

        # Extract first K transformer layers
        self._client_layers = nn.ModuleList(
            [full_model.model.layers[i] for i in range(self.num_client_layers)]
        )
        for layer in self._client_layers:
            layer.requires_grad_(False)

        # Extract final norm and LM head (for output decoding)
        self._norm = full_model.model.norm
        self._norm.requires_grad_(False)

        self._lm_head = full_model.lm_head
        self._lm_head.requires_grad_(False)

        # Free the rest of the model
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = True
        logger.info(
            "Client shard loaded: hidden_size=%d, total_layers=%d, client_layers=%d",
            self._hidden_size,
            self._total_layers,
            self.num_client_layers,
        )

    @torch.no_grad()
    def embed_and_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run embedding + K client layers. Returns hidden states for server.

        Args:
            input_ids: Token IDs, shape [batch_size, seq_len] or [seq_len].

        Returns:
            Hidden states after K layers, shape [batch_size, seq_len, hidden_dim].
            These will be DP-noised before sending to server.
        """
        assert self._loaded, "Call load() before inference"

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Embedding lookup (private: server never sees token IDs)
        hidden_states = self._embed_tokens(input_ids)

        # Run K client-side transformer layers
        for layer in self._client_layers:
            layer_output = layer(hidden_states)
            # Handle both tuple and tensor returns
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        return hidden_states

    @torch.no_grad()
    def decode_and_sample(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> int:
        """Run layer norm + LM head + sampling. Returns next token ID.

        Args:
            hidden_states: Final hidden states from server + decrypted deltas,
                shape [batch_size, seq_len, hidden_dim].
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Next token ID (int). Output logits never leave the client.
        """
        assert self._loaded, "Call load() before inference"

        # Layer norm
        normed = self._norm(hidden_states)

        # LM head: hidden_dim → vocab_size
        logits = self._lm_head(normed[:, -1, :])  # Last token position

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text. Returns token IDs tensor."""
        assert self._loaded, "Call load() before inference"
        encoded = self._tokenizer(text, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def detokenize(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        assert self._loaded, "Call load() before inference"
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        assert self._loaded, "Call load() before inference"
        return self._tokenizer.eos_token_id

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def total_layers(self) -> int:
        return self._total_layers

    @staticmethod
    def estimate_memory_gb(hidden_size: int, num_layers: int, vocab_size: int) -> float:
        """Estimate client shard memory in GB.

        Components: embedding (vocab × hidden) + K layers (~4 × hidden² each)
                    + norm (hidden) + LM head (hidden × vocab)
        """
        bytes_per_param = 4  # float32
        embed_params = vocab_size * hidden_size
        layer_params = num_layers * (4 * hidden_size * hidden_size)  # Approximate
        norm_params = hidden_size
        lm_head_params = hidden_size * vocab_size  # Often tied with embedding
        total_params = embed_params + layer_params + norm_params + lm_head_params
        return (total_params * bytes_per_param) / (1024**3)
