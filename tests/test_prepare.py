import json

import pytest

from feln_laya.prepare import prepare, query_key


def test_migration_split_and_source_preservation(tmp_path):
    catalog = {
        "layers": [
            {
                "name": "Wells",
                "stype": "Point",
                "columns": [
                    {
                        "name": "NAME",
                        "dtype": "String",
                        "values": ["Alpha"],
                        "hints": [
                            "Make sure to use the SQL ILIKE in the WHERE clause for the values of the column NAME."
                        ],
                    }
                ],
            }
        ]
    }
    rows = [
        {
            "text": f"wells with name containing '{name}'",
            "meta": {"layers": ["Wells"], "where": [f"NAME LIKE '%{name}%'"], "relations": []},
        }
        for name in ("Alpha", "Beta", "Gamma", "Delta")
    ]
    rows.append({**rows[0], "text": "find wells whose name contains 'Alpha'"})
    cp, rp = tmp_path / "Layers.json", tmp_path / "FELN.json"
    cp.write_text(json.dumps(catalog))
    rp.write_text(json.dumps(rows))
    original = rp.read_bytes()
    out = tmp_path / "prepared"
    report = prepare(cp, rp, out, holdout=1)
    assert rp.read_bytes() == original
    assert report["source_records"] == 5
    assert report["migrated"] == 5
    held = [json.loads(line) for line in (out / "heldout.jsonl").read_text().splitlines()]
    normalized = json.loads((out / "FELN.json").read_text())
    held_keys = {query_key(row["meta"]) for row in held}
    assert len(held) == sum(query_key(row["meta"]) in held_keys for row in normalized)
    assert report["train_examples"] + len(held) == 5
    with pytest.raises(FileExistsError):
        prepare(cp, rp, out, holdout=1)


def test_migration_preserves_case_sensitive_columns(tmp_path):
    catalog = {
        "layers": [
            {
                "name": "Wells",
                "stype": "Point",
                "columns": [
                    {
                        "name": "NAME",
                        "dtype": "String",
                        "values": ["Alpha"],
                        "hints": ["Use SQL ILIKE."],
                    },
                    {
                        "name": "TAG",
                        "dtype": "String",
                        "values": ["Marker"],
                        "hints": ["Use SQL LIKE."],
                    },
                ],
            }
        ]
    }
    rows = [
        {
            "text": f"wells with name containing '{name}' and tag containing 'Marker'",
            "meta": {
                "layers": ["Wells"],
                "where": [f"name LIKE '%{name}%' AND TAG LIKE '%Marker%'"],
                "relations": [],
            },
        }
        for name in ["Alpha", "Beta"]
    ]
    cp, rp = tmp_path / "Layers.json", tmp_path / "FELN.json"
    cp.write_text(json.dumps(catalog))
    rp.write_text(json.dumps(rows))
    out = tmp_path / "prepared"
    report = prepare(cp, rp, out, holdout=1)
    assert report["migrated"] == 2
    for row in json.loads((out / "FELN.json").read_text()):
        assert "NAME ILIKE" in row["meta"]["where"][0]
        assert "TAG LIKE" in row["meta"]["where"][0]
