"""Generated supplements must not leak source queries into training."""

import json

from feln_laya.prepare import prepare, query_key


def test_generated_examples_are_audited_and_disjoint(tmp_path):
    catalog = {
        "layers": [
            {
                "name": "Wells",
                "alias": "wells",
                "stype": "Point",
                "columns": [
                    {
                        "name": "DEPTH",
                        "alias": "depth",
                        "dtype": "Double",
                        "values": ["10", "20", "30", "40"],
                    }
                ],
            }
        ]
    }
    rows = [
        {
            "text": f"wells deeper than {v}",
            "meta": {"layers": ["Wells"], "where": [f"DEPTH > {v}"], "relations": []},
        }
        for v in (10, 20, 30)
    ]
    cp, rp = tmp_path / "layers.json", tmp_path / "records.json"
    cp.write_text(json.dumps(catalog))
    rp.write_text(json.dumps(rows))
    out = tmp_path / "data"
    report = prepare(cp, rp, out, holdout=1, generated=10)
    source_keys = {query_key(r["meta"]) for r in rows}
    generated = [json.loads(line) for line in (out / "generated.jsonl").read_text().splitlines()]
    assert report["generated_examples"] == len(generated) > 0
    assert not source_keys.intersection(query_key(r["meta"]) for r in generated)
    assert (out / "rejected-generated.jsonl").exists()
