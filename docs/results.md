# Measured results (2026-09-21)

## Verification

- Twenty-one unit test cases pass, including the pre-merge review regressions. The native MLX checkpoint test is opt-in and passed on
  the newly trained `out/small` checkpoint.
- Original NorthSea FELN.json: 2,569/3,000 roundtrip with current catalog rules.
  Migrated local copy: 3,000/3,000; 431 LIKE→ILIKE changes, source unchanged.
- CUDA and native MLX produce identical candidates and acceptance decisions on
  all 200 development examples with the baseline model. Maximum decision-confidence
  difference: 0.0001. Direct Torch/MLX card and padded span logits pass tolerance 2e-4.

## Baseline: supplied examples only

Training: 2,800 requests, 156,307 cards, 2,800 span rows. Three fresh BGE-small epochs
on gc3 GPU 0 took about 194 seconds, with ~8.6 GB peak Torch allocation.
The 200 held-out requests have no identical text or normalized-query overlap with training.

| Runtime | Exact FELN | Valid candidate | Mean partial | Coverage at 0.8 | Exact among accepted | Median request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | 55% | 98.5% | 0.868 | 29% | 77.6% | 58.6 ms |
| MLX | 55% | 98.5% | 0.868 | 29% | 77.6% | 161.3 ms |

Timings include tokenization, decision scoring and composition after model loading.
These were sequential calls, not a concurrent-throughput benchmark. No SQL was executed.
The baseline evidence predates the additional multiple-primary/extra-value abstention checks;
candidate accuracy is unaffected, but acceptance coverage is not directly comparable across
those wrapper revisions.

Six manually specified diagnostic requests scored 2/6 exact. They expose unwanted subtype
filters and weak plain-layer/distance handling. They are examples, not a representative test.

## Data and model errors are distinct

Examples of supplied text/label contradictions in the development split:

- “Find gas pipelines more than 25 kilometers from all matching gas/condensate wells.”
  Gold selects `Pipelines` and `Discoveries`; the model selects the named `Wells`.
- “List oil/condensate wells with formation tops that lie within one or more oil pipelines.”
  Gold uses `Discoveries` as the secondary, despite the explicit pipeline wording.

Those contradictions lower exact match without necessarily indicating the wrong semantic
interpretation. They do **not** explain away the genuine errors: the baseline inserts a
water-well subtype for “water depth”, reverses some date comparisons, and confuses distances
when two secondary layers use different values/units. Source text and gold were preserved
so these can be audited independently.

The expanded training recipe adds explicit-layer and subtype-free examples to address
that coverage gap. Its results are recorded below. Because the original
holdout informed this change, it is now a development set; it is not an untouched final test.

## Expanded model: final result

Recommended checkpoint: `out/expanded-small`; state/data snapshot: `out/expanded-v2`.
Training consumed 5,657 requests / 311,438 cards for three epochs on gc3 GPU 0:
488.6 seconds total, 11.3 GB peak Torch allocation, epoch losses 0.2881 / 0.0716 / 0.0444.
No original text or label was rewritten beyond the audited LIKE→ILIKE migration.

| Runtime | Exact FELN | Valid candidate | Mean partial | Coverage at 0.8 | Exact among accepted | Median request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | 74% | 99.5% | 0.934 | 53.5% | 91.6% | 58.5 ms |
| MLX | 74% | 99.5% | 0.934 | 53.5% | 91.6% | 161.5 ms |

CUDA/MLX candidates and statuses match on all 200 development examples plus all six
manual diagnostics. The final native card/span parity test passes. The manual requests
improved from 2/6 to 5/6 exact, including the 7-kilometer and 37-meter relationships and
`EXAMPLE ENERGY` string. These six examples are diagnostic, not an independent benchmark.

**A confident error remains:** “Find wells with water depth greater than 1234 meters”
adds `content_type = 14` (water wells), although no subtype was requested. It passes the
0.8 gate at confidence 0.9242. A high decision margin does not guarantee the full query is
correct. Two correctly translated distance examples abstain. This is an experimental
baseline, not reliable unattended query execution.

Files under `results/` preserve both model runs, per-request predictions, aggregate metrics,
training logs, environment versions and model checksums. `expanded-backend-comparison.json`
records the final equality check. This evaluation measures structure, not selected GIS
feature IDs or spatial execution correctness.
