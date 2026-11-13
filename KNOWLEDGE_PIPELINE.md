# Knowledge-RAGPT/PER pipeline

This checkout keeps the original ComP/PER forward and CLI intact and adds an
opt-in, train-only structured-knowledge path.

## Pipeline

```text
observed features + raw text
→ target-specific EvidenceQuery
→ clean train-only context retrieval
→ schema-constrained rule knowledge card
→ deterministic card feature encoder
→ target-specific card projector
→ existing PER EvidenceEncoder and ComP fusion
```

No test split is read by the build, training, cache, filter or validation
commands. `oracle_card` is validation-only. Test evaluation additionally
requires `--allow-test`.

## Environment

```bash
source /home/xuanqin/.venvs/pytorch/bin/activate
export PEROUTER_DATASET_ROOT=/home/xuanqin/Comp/datasets/ComP_clean
export PEROUTER_SAVED_ROOT=/home/xuanqin/PERouter/runs/comP_saved
```

The environment defaults still point to the original server paths, so setting
these variables is only necessary on another machine.

## One-shot validation pipeline

First select or create a frozen fair B0 checkpoint:

```bash
PYTHON=/home/xuanqin/.venvs/pytorch/bin/python \
GPU=0 \
SEED=66 \
bash scripts/bootstrap_b0_for_knowledge.sh
```

The command prints the checkpoint and validation-manifest paths. Then run:

```bash
export STUDENT_CHECKPOINT=/path/printed/by/bootstrap
export VAL_MASK_MANIFEST=/path/printed/by/bootstrap
export RUN_ROOT=runs/knowledge_ragpt/full_seed66

PYTHON=/home/xuanqin/.venvs/pytorch/bin/python \
GPU=0 \
bash scripts/run_full_knowledge_pipeline.sh
```

The full pipeline performs:

1. train-only context index;
2. knowledge-card generation and schema validation;
3. card-projector training;
4. knowledge candidate cache generation;
5. listwise and pairwise cross-fit filter training;
6. empty/raw/real/random/shuffled/source-only/confidence/random-action evaluation;
7. learned-filter evaluation;
8. validation Oracle card;
9. aggregate metrics, gate decisions and closure hashes.

## Main artifacts

```text
RUN_ROOT/
├── knowledge_index/
├── knowledge_cards/
├── adapter/
├── filter_training/
├── validation/
└── closures/
```

## Tests

```bash
python -m pytest -q
```

The local CPU verification completed with 120 passed and 3 environment-gated
skips. A synthetic CMU-MOSI-format run completed the entire one-shot pipeline
successfully; those synthetic metrics are functional evidence only and must
not be interpreted as MOSI research results.
