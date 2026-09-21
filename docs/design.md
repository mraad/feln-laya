# Design and limits

## Reuse before new code

The requested references contain complementary pieces:

| Reference | Used here |
| --- | --- |
| `../feln` | FELN shape, relation semantics, normalization and comparison |
| `../layers-json` | Typed project catalog, aliases, codes, units and hints |
| `../laya` | Runtime-defined choice/noul decision pattern and batching |
| `../decider` | Explicit probability-bearing typed decisions |
| `../SemIf` | Native MLX scoring rather than text generation; parity checks |
| `../feln-type` | Existing catalog-grounded question grammar, span head, composer and training |

The adjacent `feln-type` implementation already solves the difficult query decomposition
problem. Reusing it is smaller and better tested than copying its SQL grammar or adding
another language model. This project owns state/checkpoint consistency, stricter abstention,
length-aware batching, MLX inference, isolated data preparation, CLI and evaluation evidence.

No HTTP server, general Score endpoint, retrieval database, or distributed training is added.
FELN uses categorical and boolean decisions; it does not need an ordinal Score. A 33M BERT
checkpoint fits on one GPU. Additional GPUs are not needed for this baseline.

## MLX

`mlx_backend.py` loads BERT safetensors and the two trained heads. It implements the standard
embedding sum, post-normalized attention and feed-forward blocks, exact GELU, padding mask,
CLS card head and token-wise BIO span head. Native fused scaled-dot-product attention and
layer normalization keep the implementation short. Dropout is inactive at inference.

MLX is float32 to preserve the existing checkpoint's behavior. Quantization would require
new parity and accuracy evidence and is not implemented. Variable-length claims are sorted
before minibatching, so padding work is reduced without changing decision order. Tokenization
and small probability/composition operations stay on CPU. The current adapter retains Torch
as a dependency to reuse the upstream pipeline, even on MLX.

## Current project state

On 2026-09-21 the live NorthSea catalog hash was `890d2c62263b3c4d`; previously trained
checkpoints used `da07bdf016fbb33d`. Catalog changes included LIKE→ILIKE rules, water-depth
units/descriptions, and aliases. Original FELN.json had 431 case-sensitivity mismatches.
Preparation migrated those filters in a local copy only. All 3,000 copied examples roundtrip.
The audit records before/after metadata and source-file checksum.

The 2,800 training and 200 held-out examples have disjoint normalized query keys (including
canonical relation units and secondary order). This is a same-project evaluation; it does
not demonstrate transfer to unseen catalogs, unseen SQL grammars, or arbitrary prose.
Training starts from the cached pretrained BGE-small encoder and uses 156,307 sampled cards
and 2,800 span-label rows for three epochs. No held-out examples enter training.

The first model added unwanted subtypes to plain-layer requests. A second recipe adds
2,857 roundtrip-checked catalog-generated requests with explicit aliases and a 50%
layer-only sampling rate. It contains 5,657 training requests / 311,438 cards and keeps
the same 200 original held-out requests. Twenty-six generated rows outside the shared
grammar are audited and excluded; source queries are excluded from generated additions.
This holdout is now a development set used to compare recipes, not an untouched final test.

## Boundaries

The inherited grammar supports the FELN operators in `feln-type.cards`, not unrestricted SQL.
It has one operator per field, bounded AND/OR shapes and literal-span requirements. The
composer may choose the wrong valid query; validation alone does not establish correctness.
This wrapper rejects missing literals and conflicting literal ownership rather than accepting
an incomplete filter. It also rejects a forced primary when the role decision is inconsistent.

No live database resolver is enabled. String literals come from the request and catalog rules;
they are not checked against current database rows. SQL is never executed. Scores are domain
model decision scores, not calibrated end-to-end probabilities. The threshold must be assessed
on a representative validation set before building an automatic execution workflow.

State and example files may change after training. Pass the training snapshot to reproduce a
run, or prepare and retrain for the live catalog. Do not bypass the hash check by editing a
checkpoint's config. The original NorthSea and all reference checkouts remain unchanged.
