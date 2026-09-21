"""Prepare a reproducible copy; never rewrite source FELN.json or its catalog."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from feln import FELN
from feln.compare import canonical_text
from feln.generate import generate
from feln_type.data import gold_spans, load_records, rows_from
from feln_type.decompose import compose, decompose, gold_answers
from feln_type.state import catalog_sha
from layers_json.layers import Layers


def query_key(meta):
    return canonical_text(FELN(**meta))


def prepare(catalog_path, examples_path, output, holdout=200, seed=0, generated=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    catalog = Layers.load(catalog_path)
    records = load_records(str(examples_path))
    normalized, changes = [], []
    for i, row in enumerate(records):
        gold = FELN(**row["meta"])
        ex = decompose(catalog, row["text"], gold)
        composed = compose(catalog, ex, gold_answers(ex))
        if composed.feln is None:
            raise ValueError(f"Cannot compose row {i}: {composed.reason}")
        # Only migrate LIKE -> ILIKE where current catalog requires it. Reject other drift.
        from feln.identical import parse_where
        from sqlglot import exp

        migrated = []
        for old in gold.where:
            tree = parse_where(old) if old else None
            if tree is not None:
                tree = tree.transform(
                    lambda n: exp.ILike(**n.args) if isinstance(n, exp.Like) else n
                )
            migrated.append(tree.sql(dialect="duckdb") if tree is not None else "")
        allowed = FELN(layers=gold.layers, where=migrated, relations=gold.relations)
        if not composed.feln.same(gold) and not composed.feln.same(allowed):
            raise ValueError(f"Row {i} has unsupported drift; inspect before training")
        if not composed.feln.same(gold):
            changes.append({"index": i, "before": row["meta"], "after": composed.feln.model_dump()})
        normalized.append({**row, "meta": composed.feln.model_dump()})
    groups = defaultdict(list)
    for row in normalized:
        groups[query_key(row["meta"])].append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    held, train = [], []
    for key in keys:
        (held if len(held) < holdout else train).extend(groups[key])
    if not train or not held:
        raise ValueError("Need nonempty train and held-out groups")
    (output / "Layers.json").write_bytes(Path(catalog_path).read_bytes())
    (output / "FELN.json").write_text(json.dumps(normalized, ensure_ascii=False))
    (output / "migrations.json").write_text(json.dumps(changes, indent=2))
    # Fill the subtype-free request gap without duplicating any source query.
    synthetic, rejected = [], []
    for row in generate(catalog, generated, seed=seed + 1, alias_suffix=True, layer_only=0.5):
        if query_key(row["meta"]) in groups:
            continue
        try:
            ex = decompose(catalog, row["text"], FELN(**row["meta"]))
            check = compose(catalog, ex, gold_answers(ex))
            if check.feln is None or not check.feln.same(ex.gold):
                raise ValueError("Generated example failed roundtrip")
        except ValueError as error:
            rejected.append({**row, "reason": str(error)})
            continue
        synthetic.append(row)
    (output / "generated.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synthetic))
    (output / "rejected-generated.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rejected)
    )
    train.extend(synthetic)
    rows, spans = [], []
    for row in train:
        ex = decompose(catalog, row["text"], FELN(**row["meta"]))
        rows.extend(asdict(r) for r in rows_from(ex, rng, negatives=4))
        spans.append({"text": ex.text, "spans": gold_spans(ex)})
    rng.shuffle(rows)
    for name, records in (("train_rows", rows), ("train_spans", spans), ("heldout", held)):
        (output / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in records))
    meta = dict(
        catalog_sha=catalog_sha(catalog),
        train_examples=len(train),
        train_rows=len(rows),
        heldout=len(held),
        seed=seed,
        augmentation=False,
        generated_examples=len(synthetic),
        rejected_generated=len(rejected),
    )
    meta.update(
        source_sha256=hashlib.sha256(Path(examples_path).read_bytes()).hexdigest(),
        migrated=len(changes),
        source_records=len(normalized),
        split="disjoint normalized FELN query groups; seed " + str(seed),
    )
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalog", type=Path)
    p.add_argument("examples", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--generated", type=int, default=0, help="additional catalog-generated training examples"
    )
    args = p.parse_args()
    if args.generated < 0:
        p.error("--generated must be nonnegative")
    print(
        json.dumps(
            prepare(args.catalog, args.examples, args.output, generated=args.generated), indent=2
        )
    )


if __name__ == "__main__":
    main()
