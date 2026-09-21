"""Opt-in real checkpoint parity: FELN_TEST_MODEL=/path/to/model pytest tests/test_mlx.py."""

import os

import pytest
import torch
from feln_type.model import collate

from feln_laya.engine import BatchedPredictor


@pytest.mark.skipif(
    not os.environ.get("FELN_TEST_MODEL"), reason="set FELN_TEST_MODEL for native GPU test"
)
def test_native_mlx_matches_torch():
    torch.set_num_threads(4)
    path = os.environ["FELN_TEST_MODEL"]
    pt = BatchedPredictor(path, device="cpu", batch=2)
    mx = BatchedPredictor(path, backend="mlx", batch=2)
    text = "Find wells deeper than 350 meters within 5 miles of pipelines operated by Equinor."
    claims = [
        "layer Wells (wells, Point): primary",
        "layer Pipelines: unused",
        "layer Wells column water_depth: more than",
        "layer Pipelines: distance 5",
    ]
    torch.testing.assert_close(
        pt.logits(text, claims), mx.logits(text, claims), atol=2e-4, rtol=2e-4
    )
    encoded = pt.tok([text, "gas wells"], padding=False)
    items = [{k: v[i] for k, v in encoded.items()} for i in range(2)]
    batch = collate(items, pt.tok.pad_token_id, torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(
            pt.model.span_logits(**batch), mx.model.span_logits(**batch), atol=2e-4, rtol=2e-4
        )
    assert pt.spans(text) == mx.spans(text)
