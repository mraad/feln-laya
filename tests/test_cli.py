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


@pytest.mark.parametrize("command", ["ask", "evaluate"])
@pytest.mark.parametrize("threshold", ["-1", "1.1", "nan", "inf", "-inf"])
def test_invalid_threshold_fails_before_loading_model(command, threshold, tmp_path, monkeypatch):
    def unexpected_engine(*args, **kwargs):
        pytest.fail("invalid threshold must be rejected before loading a model")

    output = tmp_path / "evidence.jsonl"
    argv = [
        "feln-laya",
        command,
        "catalog.json",
        "input",
        "--model",
        "model",
        f"--threshold={threshold}",
    ]
    if command == "evaluate":
        argv += ["--output", str(output)]
    monkeypatch.setattr(cli, "Engine", unexpected_engine)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not output.exists()
