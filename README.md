# FELN Laya

Project state + natural-language request → typed decisions → a validated FELN query.
Runs on CUDA, Apple MPS, CPU, or **native MLX**. There is no generated JSON to repair.
The current implementation uses `Layers.json`; OKF is not needed when the typed catalog
is available.

This is a domain-specific Jev-like decision pipeline, not a reproduction of Jev's model
or a general-purpose Jev API server. It reuses the local `feln-type` grammar, trained
cross-encoder and span head, and `feln` validation/comparison. Laya and decider informed
the choice/noul interface; SemIf informed native MLX inference and parity checking.
No hosted inference service is used.

## Install

Keep `../feln-type`, `../feln`, and `../layers-json` beside this checkout.

```sh
uv sync --extra dev              # CUDA / MPS / CPU
uv sync --extra dev --extra mlx  # Apple Silicon, native MLX
```

`feln` currently pins its catalog dependency to the public `layers-json` commit recorded
in `uv.lock`. The editable `feln-type` and `feln` source checkouts are runtime dependencies,
not copied into this repository. Model weights are separate, under `out/`.

## Translate a request

On the development machine, the trained current-state checkpoint is `out/expanded-small`.
Weights, prepared data, and detailed inference dumps are local artifacts excluded from Git;
a fresh checkout must run preparation and training first:

```sh
uv run --extra mlx feln-laya ask out/expanded-v2/Layers.json \
  'Find gas/condensate wells.' --model out/expanded-small --backend mlx
```

Measured on 200 development requests: **74% exact FELN match**. At threshold 0.8,
53.5% are accepted and 91.6% of those match exactly. This remains an experimental
translator with known confident errors; see [measured results](docs/results.md).

To reproduce preparation and training, use fresh output directories:

```sh
N="$HOME/Documents/ArcGIS/Projects/NorthSea"
uv run python -m feln_laya.prepare "$N/Layers.json" "$N/FELN.json" out/my-data --generated 3000
uv run feln-laya roundtrip out/my-data/Layers.json out/my-data/FELN.json
uv run feln-laya train out/my-data out/my-model --device mps
```

Use `--device cuda` for Torch CUDA and `CUDA_VISIBLE_DEVICES=0` to select one GPU.
The MLX backend runs the BERT encoder and both heads on the Apple GPU; Torch only reads
saved head weights and handles the small CPU tensors used by the shared composer.
It does not load the Torch encoder. MLX currently supports the BERT checkpoints produced
by this training recipe, in float32; it does not load arbitrary Laya or Qwen weights.

`prepare` keeps the source files unchanged. It writes a copied catalog, normalized examples,
a migration audit, training cards/span labels, and a held-out split. It only permits the
observed LIKE→ILIKE catalog migration; other changes fail for inspection. Equivalent
normalized FELN queries stay in the same split. Optional `--generated 3000` supplements
training with catalog-generated requests that
explicitly name layers and often omit subtype filters. Generated rows outside the grammar
are saved in `rejected-generated.jsonl`; none of the original examples are silently dropped.
Original query groups are excluded from this supplement. Literal-swapping augmentation is omitted.
`train` starts from bge-small, not a checkpoint that already saw the held-out examples.
It saves weights and `run.json` after each epoch and refuses an existing output directory.

An answer includes:

- `meta`: the FELN query only when accepted; `null` on abstention.
- `candidate`: the diagnostic proposal, including on abstention.
- `answers`: active choice/noul decisions with probabilities and margins.
- `confidence`: the minimum active decision margin, **not** the probability that the
  complete query is correct. The default threshold is 0.8 and is not a safety guarantee.
- `catalog_sha`, timing, backend, and abstention reasons.

For example, the grammar represents “gas/condensate wells” on NorthSea as:

```json
{"layers":["Wells"],"where":["content_type = cast(7 as SMALLINT)"],"relations":[]}
```

The model chooses layers, operators, relation types, and literal spans. The catalog supplies
column meanings and codes; the shared composer renders SQL predicates and relation strings.
Inference scores similar-length cards together to reduce padding, then restores option order.
It rejects oversized requests instead of silently truncating them, and refuses checkpoints
trained against a different catalog hash. Missing values and competing field assignments
cause abstention. Queries are returned as data; **this package does not execute SQL**.

```python
from feln_laya import Engine

engine = Engine("out/expanded-v2/Layers.json", "out/expanded-small", backend="mlx")
result = engine.ask("Find wells within 5 kilometers of pipelines.")
if result["status"] == "accepted":
    print(result["meta"])
```

## Evaluate and resume work

```sh
uv run pytest -q
FELN_TEST_MODEL=out/expanded-small uv run --extra mlx pytest tests/test_mlx.py -q
uv run --extra mlx feln-laya evaluate out/expanded-v2/Layers.json out/expanded-v2/heldout.jsonl \
  --model out/expanded-small --backend mlx --output results/my-mlx-heldout.jsonl
uv run ruff check .
uv run ruff format --check .
```

Evaluation writes and flushes one JSONL result per example, then a `.summary.json` with exact
`FELN.same`, structural similarity, validity, acceptance coverage/accuracy, and timings.
It refuses to overwrite evidence. Partial JSONL files survive interruptions; use a new
output path for another run. No execution accuracy is claimed. The held-out labels come
from humanized synthetic examples and can themselves contain wording errors.

The [gc3 runbook](docs/gc3.md) records the isolated remote training setup.
Read [PROGRESS.md](PROGRESS.md) first after a disconnect. It tracks completed checks,
training location, pending permissions, and the next step. [docs/design.md](docs/design.md)
explains the implementation limits and reference choices.

Git includes aggregate metrics and training logs under `results/`. Detailed per-request
JSONL evidence and the full CUDA smoke response remain local.
