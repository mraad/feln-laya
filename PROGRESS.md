# Progress / resume notes

## 2026-09-21 — discovery

Request: a simple Jev-like implementation using local references, project state
(`Layers.json` or OKF), and `FELN.json` examples; test locally with MLX and/or on gc3.

- Workspace initially contained only AGENTS.md, no git repository or code.
- Read references: ../feln (FELN validation/comparison/SQL), ../layers-json
  (typed catalog), ../laya (choice/score/noul), ../decider (typed API),
  ../SemIf (native MLX option-logit scoring).
- NorthSea: /Users/mraad/Documents/ArcGIS/Projects/NorthSea, containing
  Layers.json, OKF, and 3,000 humanized text/meta examples in FELN.json.
- Found an existing adjacent ../feln-type implementation of atomic decisions
  and composition, with trained cross-encoders. Inspect before duplicating it.
- gc3 SSH works with network escalation. Four RTX PRO 6000 GPUs; GPUs 0–2
  were idle, GPU 3 occupied. Do not interfere with other jobs.
- Asked whether the main interface should translate text into new FELN or choose
  supplied candidates. Awaiting answer while inspecting reusable components.

No project data or reference source has been modified. No models trained yet.

## Implementation and catalog drift

- Added Engine.ask, Torch/CUDA inference, native MLX BERT inference, CLI,
  roundtrip and flushed JSONL evaluation, and boundary regression tests.
- Reusing ../feln-type as an editable dependency (grammar, composer, training);
  no new generative parser. Laya/decider supply the typed-decision pattern.
- Existing checkpoints use catalog hash da07bdf016fbb33d; live catalog is
  890d2c62263b3c4d. The matching old snapshot is NorthSea/backups/
  regen-20260921-150237-UTC/Layers.json. Changes include LIKE→ILIKE,
  descriptive hints, water-depth units, and aliases.
- Original FELN.json roundtrip: 2,569/3,000; 431 LIKE→ILIKE mismatches.
  Source data is unchanged. prepare.py creates out/data with audited migrations
  and a split grouped by normalized FELN, plus original-source checksum.
- Fresh training planned on gc3 GPU 0. Local MLX needs unsandboxed execution
  because the sandbox cannot see Metal. Dependency setup completed.
- Boundary tests: 3 passed. MLX smoke correctly rejected the stale checkpoint;
  numerical parity and fresh-model inference remain pending.

## Current training / resume checkpoint

- Original source files remain unchanged. `out/current/` is the successful prepared
  dataset: 2,800 training examples, 156,307 cards, 2,800 span rows; 200 held out.
  `out/data/` is an earlier incomplete preparation attempt; do not use it.
  Upstream synthetic augmentation hit a roundtrip failure, so the simpler final
  preparation uses original humanized records only, without augmentation.
- Roundtrip on out/current/FELN.json: **3,000/3,000**.
- Unit tests: **6 passed**, native MLX test skipped unless FELN_TEST_MODEL is set.
- Explicit native MLX parity test using ../feln-type/out/small: **passed**,
  both card logits and padded BIO span logits agree with Torch at atol/rtol 2e-4.
- Automatic approval review rejected uploading NorthSea-derived data to gc3.
  Explicit user permission was requested asynchronously and has not arrived.
  Do not upload the archive/data without that approval.
- Training is running **locally on MPS**, exec session 2816, output `out/small`:
  BGE-small, 3 epochs, batch 128, 156,307 cards, 2,800 span rows, seed 0.
  Epoch 1 at step 200/1222: loss 0.7792, 431 rows/sec.
  The trainer saves weights/config/run.json after each epoch.
- Next: finish training; run native MLX parity on out/small; evaluate heldout
  with MLX; record measured accuracy/coverage/timings in docs. CUDA smoke may
  use data/checkpoints already resident on gc3 without new project-data transfer.

## CUDA validation and tests

- CUDA smoke **passed** on gc3 GPU 0 using only the catalog/checkpoint already
  resident there. Uploaded implementation source and synthetic tests only; no
  NorthSea catalog or derived training examples were uploaded.
- Remote directory: `/home/ubuntu/feln-laya-cuda-smoke`.
  Existing checkpoint: `/home/ubuntu/feln-type/out/gc3-small`.
  Existing catalog: `/home/ubuntu/data/northsea/Layers.json` (old hash da07bdf016fbb33d).
- Request: Find wells within 5 kilometers of pipelines.
  Candidate correct: Wells/Pipelines, empty filters, withinDistance 5 kilometers.
  Confidence 0.1179, status abstain at threshold 0.8, 0.365 s inference.
  This is a runtime smoke, not evidence about the fresh current-catalog model.
- Unit tests now **7 passed, 1 opt-in native test skipped**. The explicit native
  test previously passed. Train/heldout identical-text overlap is zero.
- README.md and docs/design.md now document setup, contracts, MLX details,
  migration, evaluation, and grammar limits. Local 3-epoch training continues.

Training heartbeat: epoch 1 step 600/1222, loss 0.4344, ~313 rows/sec;
first checkpoint not yet saved. Session 2816 remains active. Validation summaries
are saved under results/; current-state model evaluation is still pending.

User clarified the intended interface: **translate natural-language requests into
new FELN queries**, rather than choosing from supplied complete-query candidates.
The current implementation follows this scope.

## Remote training authorized and started

The user explicitly approved copying NorthSea catalog and derived examples to gc3.
The prior upload block is resolved. Copied source plus out/current to isolated
`gc3:/home/ubuntu/feln-laya-20260921` and started:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
/home/ubuntu/feln-type/.venv/bin/python -u -m feln_laya.cli \
train out/current out/small --device cuda --batch 256
```

Remote log: `/home/ubuntu/feln-laya-20260921/train.log`.
Launcher PID reported 2972464. Check `out/small/run.json` for saved epochs.
Stopped duplicate local MPS training PID 30252 with SIGINT. No local completed
checkpoint was used. Next: finish remote three epochs, evaluate on CUDA, download
out/small, then run native MLX parity and held-out evaluation locally.

Remote provenance check: model.py, predict.py and cards.py match local SHA256;
decompose.py differs in the older remote checkout. Training consumes already-built
cards and the identical model.py, so this does not affect training. Copied the local
feln_type package into the isolated remote work directory for matching evaluation;
the shared `/home/ubuntu/feln-type` checkout remains unchanged.

## Baseline complete; training coverage issue found

- Fresh CUDA training completed: 3 epochs, losses 0.3583 / 0.1376 / 0.1033,
  ~64 seconds/epoch, ~2,450 rows/sec, ~8.6 GB peak allocation.
- Downloaded checkpoint to local out/small. Current-checkpoint native MLX parity
  test passed. Full 200-record MLX and CUDA candidates and statuses agree 200/200;
  maximum confidence difference 0.0001.
- Baseline heldout: exact 0.55, valid 0.985, mean partial 0.86785; threshold 0.8
  accepts 29%, accepted accuracy 0.77586. Median CUDA 0.0586 s, MLX 0.1613 s.
  Evidence: results/{cuda,mlx}-heldout.jsonl and summary files.
- Six additional diagnostic requests (examples/northsea.json): 2/6 exact.
  The model adds unwanted subtypes to plain-layer requests (e.g. water depth →
  water wells). This prompted a generic training-coverage improvement, not
  hand-written fixes to those examples.
- Some supplied labels contradict their text. First heldout miss says
  gas/condensate wells but gold selects Discoveries; another says oil pipelines
  but gold selects Discoveries. Keep these visible; do not silently relabel them.
- prepare.py now supports --generated to add catalog-generated examples with
  explicit layer aliases and layer_only=0.5. It excludes all original query keys,
  keeps the original split, roundtrip-checks additions, and audits unsupported
  generated rows separately. Source examples still fail loudly on unsupported drift.
- Current preparation: out/expanded-v2 --generated 3000 (session 82618).
  out/expanded is an incomplete attempt: generator can emit subtype-inside-OR,
  outside the inherited grammar. Such generated rows are now recorded/rejected.
- Next: upload expanded-v2; train out/expanded-small on gc3; reevaluate unchanged
  heldout and diagnostic requests on CUDA/MLX; document final measured limits.
  The 200-row split is now a development holdout used to compare recipes, not a
  pristine final test set. Preserve baseline evidence.

Expanded run launched (PID reported 2980131):
`gc3:/home/ubuntu/feln-laya-20260921/out/expanded-small`, log `expanded-train.log`.
Dataset out/expanded-v2 has 5,657 examples, 311,438 cards; 2,857 generated added,
26 generated rejected/audited. Original holdout SHA256 remains
0bfef754a79ac9cbef86cb94ba01f62e2ac663b3e7afe58da407be8057f790f1.
Training recipe unchanged: BGE-small, seed 0, three epochs, CUDA GPU 0, batch 256.

Final boundary refinements added (and uploaded to gc3): abstain on multiple
primary decisions or excess literal candidates, in addition to missing/conflicting
values. Unit suite now 9 passed, 1 opt-in native test skipped. Baseline candidate
metrics remain applicable; old acceptance metrics predate these extra checks.
Expanded training heartbeat: epoch 1 step 400/1217, ~1,888 rows/sec.

## COMPLETE — final handoff

Implemented natural language → new FELN queries with project-state grounding,
choice/noul decisions, explicit abstention, Torch/CUDA and native MLX inference.
No supplied complete-query candidate list is required.

Recommended local artifacts:
- Model: `out/expanded-small`
- Catalog and prepared data: `out/expanded-v2`
- Usage: README.md; design: docs/design.md; measured results: docs/results.md
- Remote reproduction: docs/gc3.md
- Raw evidence: results/*expanded*; model hashes: results/environment.json

Final checks:
- 9 unit tests passed; native MLX test separately passed on final checkpoint.
- Ruff lint/format passed; 3,000 source-copy roundtrips plus 2,857 generated
  additions validated before training.
- Development set (200): 74% exact, 99.5% valid, 0.934 mean partial.
  Threshold 0.8: 53.5% coverage, 91.6% exact among accepted.
- Median request: CUDA 58.5 ms; MLX 161.5 ms.
- CUDA/MLX candidates and statuses identical on all 206 evaluated requests.
- Six manual diagnostics: 5/6 exact. The remaining error is a confident unwanted
  water-well subtype for a water-depth filter. Documented, not patched by phrase.
- Fresh expanded training finished in 488.6 seconds on gc3 GPU 0. Model and logs
  retrieved. No training/evaluation job remains running from this task.

Original NorthSea files and all local reference checkouts are unchanged.
The gc3 data-transfer approval was explicitly granted and the earlier block resolved.
No pending permission or implementation step remains. Known model/data limitations
are documented; the original 200-example holdout is a development set after recipe
comparison, not a pristine final test. No SQL was executed.

## Pre-merge review follow-up

Checked all three CodeRabbit findings against the implementation and local composer.
Fixed invalid CLI thresholds being treated as successful per-row abstentions, and
restricted LIKE→ILIKE migration to the individual catalog column's rule (mixed
case-sensitive and case-insensitive columns now work).

The proposed global cross-layer span restriction was rejected: the actual shared
composer deduplicates within each layer and preserves both filters when one literal
applies to two layers. A real decompose/compose/Engine regression test verifies this;
the existing same-layer conflict test remains in place. No inference behavior changed.

Validation: 21 unit cases passed, 1 optional native-GPU case skipped; Ruff lint and
format checks passed. Merge is pending the updated remote review/check status.
