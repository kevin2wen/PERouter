#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${STUDENT_CHECKPOINT:?set STUDENT_CHECKPOINT to the frozen fair student checkpoint}"
: "${VAL_MASK_MANIFEST:?set VAL_MASK_MANIFEST to the frozen validation mask CSV}"

PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-runs/knowledge_ragpt/full_$(date +%Y%m%d-%H%M%S)}"
GPU="${GPU:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
FILTER_EPOCHS="${FILTER_EPOCHS:-50}"
TOP_K="${TOP_K:-5}"
GAMMA="${GAMMA:-0.5}"
MAX_BATCHES="${MAX_BATCHES:-0}"

mkdir -p "$RUN_ROOT"
echo "PIPELINE_START root=$RUN_ROOT student=$STUDENT_CHECKPOINT gamma=$GAMMA top_k=$TOP_K"
echo "HARD_GUARD split=validation test_access=disabled"

"$PYTHON" -u scripts/build_knowledge_index.py \
  --student-checkpoint="$STUDENT_CHECKPOINT" \
  --top-k="$TOP_K" \
  --output="$RUN_ROOT/knowledge_index/knowledge_index.pt"

"$PYTHON" -u scripts/build_knowledge_cards.py \
  --student-checkpoint="$STUDENT_CHECKPOINT" \
  --mask-manifest="$VAL_MASK_MANIFEST" \
  --query-split=validation \
  --top-k="$TOP_K" \
  --max-batches="$MAX_BATCHES" \
  --output-dir="$RUN_ROOT/knowledge_cards"

"$PYTHON" -u scripts/validate_knowledge_cards.py \
  --cards="$RUN_ROOT/knowledge_cards/knowledge_cards.jsonl" \
  --output="$RUN_ROOT/knowledge_cards/schema_validation.json"

"$PYTHON" -u ComP/train_knowledge.py \
  --student-checkpoint="$STUDENT_CHECKPOINT" \
  --knowledge-index="$RUN_ROOT/knowledge_index/knowledge_index.pt" \
  --val-mask-manifest="$VAL_MASK_MANIFEST" \
  --epochs="$TRAIN_EPOCHS" \
  --gamma="$GAMMA" \
  --top-k="$TOP_K" \
  --max-train-batches="$MAX_BATCHES" \
  --max-val-batches="$MAX_BATCHES" \
  --gpu="$GPU" \
  --output-dir="$RUN_ROOT/adapter"

CARD_ADAPTER="$RUN_ROOT/adapter/best_card_adapter.pt"
TRAIN_MASK_MANIFEST="$RUN_ROOT/adapter/train_mask_manifest.csv"

"$PYTHON" -u scripts/build_knowledge_candidate_cache.py \
  --student-checkpoint="$STUDENT_CHECKPOINT" \
  --card-adapter-checkpoint="$CARD_ADAPTER" \
  --knowledge-index="$RUN_ROOT/knowledge_index/knowledge_index.pt" \
  --mask-manifest="$TRAIN_MASK_MANIFEST" \
  --gamma="$GAMMA" \
  --max-batches="$MAX_BATCHES" \
  --gpu="$GPU" \
  --output="$RUN_ROOT/filter_training/knowledge_candidate_cache.pt"

for OBJECTIVE in listwise pairwise; do
  "$PYTHON" -u scripts/train_knowledge_filter.py \
    --candidate-cache="$RUN_ROOT/filter_training/knowledge_candidate_cache.pt" \
    --objective="$OBJECTIVE" \
    --epochs="$FILTER_EPOCHS" \
    --output-dir="$RUN_ROOT/filter_training/$OBJECTIVE"
done

"$PYTHON" -u scripts/run_knowledge_matrix.py \
  --student-checkpoint="$STUDENT_CHECKPOINT" \
  --card-adapter-checkpoint="$CARD_ADAPTER" \
  --knowledge-index="$RUN_ROOT/knowledge_index/knowledge_index.pt" \
  --mask-manifest="$VAL_MASK_MANIFEST" \
  --modes="empty,raw_feature,card,random_card,shuffled_card,source_only,confidence,random_action" \
  --split=validation \
  --gamma="$GAMMA" \
  --max-batches="$MAX_BATCHES" \
  --gpu="$GPU" \
  --output-root="$RUN_ROOT/validation" \
  --execute

for OBJECTIVE in listwise pairwise; do
  "$PYTHON" -u ComP/eval_knowledge.py \
    --student-checkpoint="$STUDENT_CHECKPOINT" \
    --card-adapter-checkpoint="$CARD_ADAPTER" \
    --knowledge-index="$RUN_ROOT/knowledge_index/knowledge_index.pt" \
    --mask-manifest="$VAL_MASK_MANIFEST" \
    --mode="${OBJECTIVE}_filter" \
    --reranker-dir="$RUN_ROOT/filter_training/$OBJECTIVE" \
    --split=validation \
    --gamma="$GAMMA" \
    --max-batches="$MAX_BATCHES" \
    --gpu="$GPU" \
    --output-dir="$RUN_ROOT/validation/${OBJECTIVE}_filter"
done

"$PYTHON" -u ComP/eval_knowledge.py \
  --student-checkpoint="$STUDENT_CHECKPOINT" \
  --card-adapter-checkpoint="$CARD_ADAPTER" \
  --knowledge-index="$RUN_ROOT/knowledge_index/knowledge_index.pt" \
  --mask-manifest="$VAL_MASK_MANIFEST" \
  --mode=oracle_card \
  --split=validation \
  --gamma="$GAMMA" \
  --max-batches="$MAX_BATCHES" \
  --gpu="$GPU" \
  --output-dir="$RUN_ROOT/validation/oracle_card"

"$PYTHON" -u scripts/aggregate_knowledge_results.py \
  --root="$RUN_ROOT/validation" \
  --output="$RUN_ROOT/validation/results.csv"

"$PYTHON" -u scripts/write_knowledge_closure.py \
  --repo="$REPO_ROOT" \
  --root="$RUN_ROOT" \
  --output="$RUN_ROOT/closures/KNOWLEDGE_CLOSURE.json"

echo "PIPELINE_DONE root=$RUN_ROOT"
echo "REVIEW validation/results.csv and validation/ek3_gate.json before any test access"
