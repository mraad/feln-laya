"""Native MLX BERT inference for existing feln-type checkpoints (no generation).

Torch only reads the saved heads and transports CPU arrays to the shared composer.
The encoder, attention, layer norms and both learned heads execute in MLX.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch


class MLXBert:
    def __init__(self, path):
        import mlx.core as mx

        if not mx.metal.is_available():
            raise RuntimeError("MLX requires an Apple Silicon GPU")
        mx.set_cache_limit(256 * 1024 * 1024)
        path = Path(path)
        self.cfg = json.loads((path / "encoder/config.json").read_text())
        c = self.cfg
        if (
            c["model_type"] != "bert"
            or c["hidden_act"] != "gelu"
            or c.get("position_embedding_type", "absolute") != "absolute"
            or c.get("is_decoder")
            or c.get("add_cross_attention")
        ):
            raise ValueError(
                "Only encoder BERT with absolute positions and exact GELU is supported"
            )
        self.weights = mx.load(str(path / "encoder/model.safetensors"))
        heads = torch.load(path / "heads.pt", map_location="cpu", weights_only=True)
        self.temperature = float(heads["temperature"].item())
        for name in ("card_head", "span_head"):
            self.weights.update(
                {f"{name}.{k}": mx.array(v.numpy()) for k, v in heads[name].items()}
            )
        mx.eval(self.weights)

    def hidden(self, input_ids, attention_mask, token_type_ids=None):
        import mlx.core as mx
        from mlx import nn

        w, c = self.weights, self.cfg
        ids = mx.array(input_ids.numpy())
        types = mx.zeros_like(ids) if token_type_ids is None else mx.array(token_type_ids.numpy())
        mask = mx.array(attention_mask.numpy())[:, None, None, :].astype(mx.bool_)

        def linear(x, key):
            return x @ w[key + ".weight"].T + w[key + ".bias"]

        def norm(x, key):
            return mx.fast.layer_norm(x, w[key + ".weight"], w[key + ".bias"], c["layer_norm_eps"])

        x = (
            w["embeddings.word_embeddings.weight"][ids]
            + w["embeddings.position_embeddings.weight"][mx.arange(ids.shape[1])]
            + w["embeddings.token_type_embeddings.weight"][types]
        )
        x = norm(x, "embeddings.LayerNorm")
        batch, length, width = x.shape
        heads = c["num_attention_heads"]
        for i in range(c["num_hidden_layers"]):
            key = f"encoder.layer.{i}."
            q, k, v = [
                linear(x, key + "attention.self." + name)
                .reshape(batch, length, heads, width // heads)
                .transpose(0, 2, 1, 3)
                for name in ("query", "key", "value")
            ]
            attn = (
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=1 / math.sqrt(width // heads), mask=mask
                )
                .transpose(0, 2, 1, 3)
                .reshape(batch, length, width)
            )
            x = norm(
                x + linear(attn, key + "attention.output.dense"), key + "attention.output.LayerNorm"
            )
            ff = nn.gelu(linear(x, key + "intermediate.dense"))
            x = norm(x + linear(ff, key + "output.dense"), key + "output.LayerNorm")
        return x

    def _logits(self, head, **enc):
        import mlx.core as mx

        x = self.hidden(**enc)
        if head == "card_head":
            x = x[:, 0]
        out = x @ self.weights[head + ".weight"].T + self.weights[head + ".bias"]
        mx.eval(out)
        return torch.from_numpy(np.array(out))

    def card_logits(self, **enc):
        return self._logits("card_head", **enc).squeeze(-1)

    def span_logits(self, **enc):
        return self._logits("span_head", **enc)
