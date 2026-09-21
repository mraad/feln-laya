# gc3 runbook

The user authorized transferring NorthSea catalog and derived examples on 2026-09-21.
The isolated run directory is `/home/ubuntu/feln-laya-20260921`. The existing
`/home/ubuntu/feln-type/.venv/bin/python` supplies CUDA dependencies. The local
`feln_type` source was copied into the isolated run directory for matching evaluation;
the pre-existing reference project was not modified.

Training command, run from that directory:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
  /home/ubuntu/feln-type/.venv/bin/python -u -m feln_laya.cli \
  train out/current out/small --device cuda --batch 256
```

The actual job uses `nohup`, writes `train.log`, and saves `out/small/run.json` after each
epoch. Inspect both after a disconnect. Do not launch another writer into `out/small`.
The CLI refuses an existing model directory; an interrupted run can be evaluated from its
last completed epoch or restarted in a new directory. Optimizer-state resume is not implemented.

Evaluate the held-out split:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
  /home/ubuntu/feln-type/.venv/bin/python -m feln_laya.cli \
  evaluate out/current/Layers.json out/current/heldout.jsonl \
  --model out/small --device cuda --output results/cuda-heldout.jsonl
```

Back on the Mac, retrieve the model and evidence with `scp -r`. Use the copied model
with `--backend mlx` and the same catalog/split. Save MLX results under a distinct path.
Compare both summaries and predictions; record discrepancies rather than assuming parity.
The current training uses one idle GPU. GPU 3 had another workload and was left alone.

The final expanded run used `out/expanded-v2` and wrote `out/expanded-small`, with
`expanded-train.log`. It completed all three epochs (488.6 seconds). Final results
are `results/cuda-expanded-{heldout,examples}.jsonl` with matching summaries. Both
checkpoint and evidence have been copied back to the Mac. The original `out/small`
run remains available as the supplied-examples-only baseline.
