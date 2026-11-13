#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-66}"
EPOCHS="${EPOCHS:-300}"
STAGE_EPOCH="${STAGE_EPOCH:-150}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/knowledge_ragpt/bootstrap_b0_seed${SEED}}"

"$PYTHON" -u ComP/train_per.py \
  --protocol=fair \
  --dataset=CMUMOSI \
  --audio-feature=wav2vec-large-c-UTT \
  --text-feature=deberta-large-4-UTT \
  --video-feature=manet_UTT \
  --per-mode=off \
  --evidence-mode=empty \
  --coordinator-mode=on \
  --seed="$SEED" \
  --batch-size=32 \
  --epochs="$EPOCHS" \
  --stage_epoch="$STAGE_EPOCH" \
  --lr=0.0001 \
  --hidden=256 \
  --depth=4 \
  --num_heads=2 \
  --drop_rate=0.5 \
  --attn_drop_rate=0.0 \
  --gpu="$GPU" \
  --mask_rate=0.7 \
  --lbd=0.3 \
  --missing-protocol=categorical \
  --top-k=5 \
  --output-dir="$OUTPUT_ROOT"

CHECKPOINT="$(find "$OUTPUT_ROOT/CMUMOSI" -path '*/fold0/best_val.pt' -print | sort | tail -n 1)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "B0_CHECKPOINT_NOT_FOUND" >&2
  exit 1
fi
echo "B0_CHECKPOINT=$CHECKPOINT"
echo "VAL_MASK_MANIFEST=$(dirname "$CHECKPOINT")/val_mask_manifest.csv"
