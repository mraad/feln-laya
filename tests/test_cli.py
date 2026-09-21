"""CLI evidence remains useful for abstentions and cannot overwrite prior runs."""

import json
import sys

import pytest

from feln_laya import cli


def test_evaluation_counts_errors_and_protects_evidence(tmp_path, monkeypatch):
    gold = {"layers": ["Wells"], "where": [""], "relations": []}
    data = tmp_path / "examples.json"
    data.write_text(json.dumps([{"text": "wells", "meta": gold}, {"text": "bad", "meta": gold}]))

    class Engine:
        sha = "test"

        def __init__(self, *args, **kwargs):
            pass

        def ask(self, text, **kwargs):
            if text == "bad":
                raise ValueError("too long")
            return {"status": "accepted", "candidate": gold, "meta": gold, "seconds": 0.1}

    output = tmp_path / "evidence.jsonl"
    monkeypatch.setattr(cli, "Engine", Engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "feln-laya",
            "evaluate",
            "catalog.json",
            str(data),
            "--model",
            "model",
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 0
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["n"] == 2 and summary["same"] == 0.5
    assert summary["coverage"] == 0.5 and summary["accepted_accuracy"] == 1
    assert len(output.read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        cli.main()
