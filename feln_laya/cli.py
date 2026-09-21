"""Small CLI with flushed per-example evidence for interrupted evaluations."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from feln import FELN, FELNCompare
from feln_type.data import load_records
from feln_type.decompose import compose, decompose, gold_answers
from layers_json.layers import Layers

from .engine import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("ask", "evaluate", "roundtrip"):
        p = sub.add_parser(command)
        p.add_argument("catalog", type=Path)
        p.add_argument("input", help="request text for ask; JSON/JSONL examples otherwise")
        if command != "roundtrip":
            p.add_argument("--model", required=True, type=Path)
            p.add_argument("--backend", choices=["torch", "mlx"], default="torch")
            p.add_argument("--device", default=None)
            p.add_argument("--batch", type=int, default=64)
            p.add_argument("--threshold", type=float, default=0.8)
        if command == "evaluate":
            p.add_argument("--limit", type=int, default=0, help="0 means all rows")
            p.add_argument("--output", type=Path, required=True, help="new JSONL evidence file")
    p = sub.add_parser("train")
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.command == "train":
        from feln_type.data import check_sha
        from feln_type.model import train

        if args.epochs < 1 or args.batch < 1:
            parser.error("epochs and batch must be positive")
        if args.output.exists():
            parser.error("training output already exists; use a new directory")
        check_sha(args.data, Layers.load(args.data / "Layers.json"))
        train(args.data, args.output, epochs=args.epochs, batch=args.batch, device=args.device)
        return 0
    if args.command == "roundtrip":
        catalog = Layers.load(args.catalog)
        rows = load_records(args.input)
        failures = []
        for i, row in enumerate(rows):
            try:
                gold = FELN(**row["meta"])
                ex = decompose(catalog, row["text"], gold)
                result = compose(catalog, ex, gold_answers(ex))
                if result.feln is None or not result.feln.same(gold):
                    failures.append({"index": i, "reason": result.reason or "mismatch"})
            except ValueError as error:
                failures.append({"index": i, "reason": str(error)})
        print(
            json.dumps({"n": len(rows), "passed": len(rows) - len(failures), "failures": failures})
        )
        return int(bool(failures))
    engine = Engine(
        args.catalog, args.model, backend=args.backend, device=args.device, batch=args.batch
    )
    if args.command == "ask":
        print(json.dumps(engine.ask(args.input, threshold=args.threshold), indent=2))
        return 0
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    rows = load_records(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        parser.error("evaluation dataset is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hits = accepted = accepted_hits = valid = 0
    partial = 0.0
    times = []
    # Exclusive creation avoids mixing runs or destroying earlier evidence.
    with args.output.open("x") as stream:
        for i, row in enumerate(rows):
            gold = FELN(**row["meta"])
            try:
                result = engine.ask(row["text"], threshold=args.threshold)
            except ValueError as error:
                result = {
                    "status": "abstain",
                    "candidate": None,
                    "reason": str(error),
                    "seconds": 0,
                }
            candidate = FELN(**result["candidate"]) if result["candidate"] else None
            hit = candidate is not None and candidate.same(gold)
            hits += hit
            valid += candidate is not None
            if candidate is not None:
                partial += FELNCompare.partial(gold, candidate)
            act = result["status"] == "accepted"
            accepted += act
            accepted_hits += act and hit
            times.append(result["seconds"])
            stream.write(
                json.dumps(
                    {"index": i, "text": row["text"], "gold": row["meta"], "same": hit, **result}
                )
                + "\n"
            )
            stream.flush()
    report = {
        "n": len(rows),
        "same": hits / len(rows),
        "valid": valid / len(rows),
        "partial": partial / len(rows),
        "coverage": accepted / len(rows),
        "accepted_accuracy": accepted_hits / accepted if accepted else None,
        "threshold": args.threshold,
        "median_seconds": statistics.median(times),
        "total_inference_seconds": sum(times),
        "backend": args.backend,
        "model": str(args.model.resolve()),
        "catalog_sha": engine.sha,
        "data_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "note": "Structural evaluation only; no SQL executed. Training overlap depends on supplied split.",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
