"""Boundary and composition regressions; no downloaded models needed."""

from types import SimpleNamespace

import pytest
import torch
from feln import FELN
from feln_type.decompose import compose, decompose, gold_answers
from feln_type.predict import GroupResult, Prediction
from layers_json.layers import Layers

from feln_laya.engine import BatchedPredictor, Engine


def test_gold_roundtrip():
    catalog = Layers.model_validate(
        {
            "layers": [
                {
                    "name": "Wells",
                    "alias": "wells",
                    "stype": "Point",
                    "columns": [
                        {"name": "DEPTH", "alias": "depth", "dtype": "Double", "values": ["1000"]}
                    ],
                },
                {"name": "Pipes", "alias": "pipes", "stype": "Polyline", "columns": []},
            ]
        }
    )
    gold = FELN(
        layers=["Wells", "Pipes"],
        where=["DEPTH > 1000", ""],
        relations=["notWithinDistance 500 meters"],
    )
    ex = decompose(catalog, "wells deeper than 1000 meters more than 500 meters from pipes", gold)
    assert compose(catalog, ex, gold_answers(ex)).feln.same(gold)


def stub_engine(confidence):
    engine = Engine.__new__(Engine)
    engine.catalog, engine.sha, engine.backend = None, "test", "test"
    gold = FELN(layers=["Wells"], where=[""], relations=[])
    pred = Prediction(
        SimpleNamespace(feln=gold, reason="", unresolved=[]),
        {"role|Wells": GroupResult("choice", "primary", {"primary": 1}, confidence)},
    )
    engine.predictor = SimpleNamespace(
        max_len=128, tok=SimpleNamespace(encode=lambda s: s.split()), predict=lambda *a: pred
    )
    return engine


def test_abstention_never_returns_actionable_meta():
    e = stub_engine(0.5)
    assert e.ask("wells")["meta"] is None
    assert e.ask("wells")["candidate"]["layers"] == ["Wells"]
    assert e.ask("wells", threshold=0.4)["status"] == "accepted"
    for text in ("", " ", "word " * 129):
        with pytest.raises(ValueError):
            e.ask(text)
    for threshold in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            e.ask("wells", threshold=threshold)


def test_sorted_batches_restore_order_and_reject_truncation():
    p = BatchedPredictor.__new__(BatchedPredictor)
    p.device, p.batch, p.max_len = torch.device("cpu"), 2, 5

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, claims, truncation):
            assert truncation is False
            return {
                "input_ids": [[int(c)] * int(c) for c in claims],
                "attention_mask": [[1] * int(c) for c in claims],
            }

    p.tok = Tokenizer()
    p.model = SimpleNamespace(card_logits=lambda **kw: kw["input_ids"][:, 0].float())
    assert p.logits("x", ["3", "1", "4", "2"]).tolist() == [3, 1, 4, 2]
    with pytest.raises(ValueError, match="exceeds"):
        p.logits("x", ["6"])
    assert p.logits("x", []).numel() == 0


def test_missing_values_and_conflicting_fields_abstain():
    from feln_laya.engine import composition_issues

    catalog = Layers.model_validate(
        {
            "layers": [
                {
                    "name": "Wells",
                    "stype": "Point",
                    "columns": [{"name": "a", "dtype": "Double"}, {"name": "b", "dtype": "Double"}],
                }
            ]
        }
    )
    pred = stub_engine(1).predictor.predict(None, "x")
    pred.groups["op|Wells|a"] = GroupResult("choice", "between", {}, 1)
    pred.groups["val|Wells|a"] = GroupResult("noul", "0:1", {"0:1": 1}, 1)
    assert "Missing value" in composition_issues(pred, catalog)[0]
    pred.groups["op|Wells|a"].choice = "gt"
    pred.groups["op|Wells|b"] = GroupResult("choice", "lt", {}, 1)
    pred.groups["val|Wells|b"] = pred.groups["val|Wells|a"]
    assert "Conflicting" in composition_issues(pred, catalog)[0]


def test_catalog_mismatch_rejected_before_model_loading(tmp_path):
    import json

    (tmp_path / "Layers.json").write_text('{"layers": []}')
    (tmp_path / "config.json").write_text(json.dumps({"catalog_sha": "stale"}))
    with pytest.raises(ValueError, match="catalog hash"):
        Engine(tmp_path / "Layers.json", tmp_path)


def test_multiple_primaries_and_extra_values_are_not_silently_dropped():
    from feln_laya.engine import composition_issues

    catalog = Layers.model_validate(
        {
            "layers": [
                {
                    "name": "Wells",
                    "stype": "Point",
                    "columns": [{"name": "DEPTH", "dtype": "Double"}],
                }
            ]
        }
    )
    pred = stub_engine(1).predictor.predict(None, "x")
    pred.groups["role|Pipelines"] = GroupResult("choice", "primary", {}, 1)
    pred.groups["op|Wells|DEPTH"] = GroupResult("choice", "gt", {}, 1)
    pred.groups["val|Wells|DEPTH"] = GroupResult("noul", "0:1,2:3", {"0:1": 1, "2:3": 1}, 1)
    reasons = composition_issues(pred, catalog)
    assert any("primary" in r for r in reasons)
    assert any("Ambiguous values" in r for r in reasons)


def test_one_literal_can_filter_two_layers_without_dropping_either_filter():
    catalog = Layers.model_validate(
        {
            "layers": [
                {
                    "name": name,
                    "stype": stype,
                    "columns": [{"name": "country", "dtype": "String", "values": ["NO", "UK"]}],
                }
                for name, stype in [("Wells", "Point"), ("Pipes", "Polyline")]
            ]
        }
    )
    text = "Find intersecting wells and pipes that are both in 'NO'."
    gold = FELN(
        layers=["Wells", "Pipes"],
        where=["country = 'NO'", "country = 'NO'"],
        relations=["intersects"],
    )
    ex = decompose(catalog, text, gold)
    answers = gold_answers(ex)
    assert answers["val|Wells|country"] == answers["val|Pipes|country"]
    assert sum(answers["val|Wells|country"].values()) == 1
    composed = compose(catalog, ex, answers)
    assert composed.feln.same(gold)
    groups = {
        g.key: GroupResult(
            g.kind,
            max(answers[g.key], key=answers[g.key].get) if answers[g.key] else None,
            answers[g.key],
            1,
        )
        for g in ex.groups
    }
    pred = Prediction(composed, groups)
    engine = stub_engine(1)
    engine.catalog = catalog
    engine.predictor.predict = lambda *args: pred
    result = engine.ask(text)
    assert result["status"] == "accepted"
    assert FELN(**result["meta"]).same(gold)
