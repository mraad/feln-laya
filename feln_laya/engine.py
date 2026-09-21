"""Reuse the tested atomic grammar; own only backend selection and the boundary."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from feln_type.cards import NO_VALUE_OPS, TWO_VALUE_OPS, column_kind
from feln_type.model import collate
from feln_type.predict import Predictor
from feln_type.state import catalog_sha
from layers_json.layers import Layers


class BatchedPredictor(Predictor):
    def __init__(self, path, backend="torch", device=None, batch=64):
        if batch < 1:
            raise ValueError("batch must be positive")
        if backend == "torch":
            super().__init__(path, device, batch)
        elif backend == "mlx":
            from transformers import AutoTokenizer

            from .mlx_backend import MLXBert

            self.device = torch.device("cpu")
            self.model = MLXBert(path)
            self.tok = AutoTokenizer.from_pretrained(Path(path) / "encoder")
            self.config = json.loads((Path(path) / "config.json").read_text())
            self.max_len = self.config["max_len"]
            self.batch = batch
        else:
            raise ValueError(f"Unknown backend: {backend}")
        if not math.isfinite(float(self.model.temperature)) or float(self.model.temperature) <= 0:
            raise ValueError("Checkpoint temperature must be finite and positive")

    @torch.no_grad()
    def logits(self, text, claims):
        if not claims:
            return torch.zeros(0)
        # Reject overflow: silently cutting a predicate changes the requested query.
        enc = self.tok([text] * len(claims), claims, truncation=False)
        pairs = [{k: enc[k][i] for k in enc} for i in range(len(claims))]
        if any(len(p["input_ids"]) > self.max_len for p in pairs):
            raise ValueError(f"Request plus catalog claim exceeds {self.max_len} tokens")
        # Similar lengths share batches, reducing padding; restore original option order.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]["input_ids"]))
        result = torch.empty(len(pairs))
        for start in range(0, len(order), self.batch):
            indices = order[start : start + self.batch]
            batch = collate([pairs[i] for i in indices], self.tok.pad_token_id, self.device)
            result[indices] = self.model.card_logits(**batch).float().cpu()
        return result


class Engine:
    def __init__(self, catalog, model, *, backend="torch", device=None, batch=64):
        self.catalog = Layers.load(catalog)
        self.sha = catalog_sha(self.catalog)
        config = json.loads((Path(model) / "config.json").read_text())
        if config.get("catalog_sha") != self.sha:
            raise ValueError(
                "Checkpoint catalog hash differs from project state; retrain for this catalog"
            )
        self.predictor = BatchedPredictor(model, backend, device, batch)
        self.backend = backend

    def ask(self, text, *, threshold=0.8):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be nonempty")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        if len(self.predictor.tok.encode(text)) > self.predictor.max_len:
            raise ValueError("Request exceeds model context")
        started = time.perf_counter()
        pred = self.predictor.predict(self.catalog, text)
        confidence = pred.min_confidence()
        issues = composition_issues(pred, self.catalog) if self.catalog is not None else []
        if pred.composed.reason:
            issues.append(pred.composed.reason)
        if confidence < threshold:
            issues.append("Decision confidence below threshold")
        if pred.composed.unresolved:
            issues.append("Unresolved values")
        accepted = pred.feln is not None and not issues
        return {
            "status": "accepted" if accepted else "abstain",
            "meta": pred.feln.model_dump() if accepted else None,
            "candidate": pred.feln.model_dump() if pred.feln else None,
            "confidence": confidence,
            "confidence_kind": "minimum active decision margin; not a calibrated query probability",
            "reason": "; ".join(issues),
            "unresolved": pred.composed.unresolved,
            "answers": {k: asdict(v) for k, v in pred.active_groups().items()},
            "catalog_sha": self.sha,
            "backend": self.backend,
            "seconds": time.perf_counter() - started,
        }


def composition_issues(pred, catalog):
    """Do not accept a query when the shared composer had to drop a predicate."""
    if pred.feln is None:
        return ["No valid FELN composed"]
    issues = []
    primary = pred.feln.layers[0]
    primaries = [
        name
        for name, role in pred.groups.items()
        if name.startswith("role|") and role.choice == "primary"
    ]
    if primaries != [f"role|{primary}"]:
        issues.append("No consistent primary-layer decision")
    for name in pred.feln.layers:
        # The composer deduplicates within a layer; one literal may filter multiple layers.
        claimed = set()
        layer = catalog.find_layer(name)
        for col in layer.columns:
            op = pred.groups.get(f"op|{name}|{col.name}")
            if op is None or op.choice in (None, "unused"):
                continue
            if column_kind(col) == "coded" or op.choice in NO_VALUE_OPS:
                continue
            values = pred.groups.get(f"val|{name}|{col.name}")
            need = 2 if op.choice in TWO_VALUE_OPS else 1
            selected = (
                sorted(
                    (k for k, p in values.probabilities.items() if p >= 0.5),
                    key=lambda k: -values.probabilities[k],
                )
                if values
                else []
            )
            if len(selected) < need:
                issues.append(f"Missing value for {name}.{col.name}")
            elif len(selected) > need:
                issues.append(f"Ambiguous values for {name}.{col.name}")
            if claimed.intersection(selected):
                issues.append(f"Conflicting value ownership on {name}.{col.name}")
            claimed.update(selected)
    return issues
