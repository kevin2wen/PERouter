from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ComP.knowledge.filter_v2.model import FilterV2Model
from train_filter_v2 import build_tensors, load_feedback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.text_model, device="cpu")
    payload, embeddings = load_feedback(args.feedback_cache, encoder, "classification")
    task = payload.get("task", "classification")
    data = build_tensors(payload["records"], embeddings, task)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FilterV2Model(384, ckpt["state_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tensors = {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        output = model(
            tensors["candidates"], tensors["mask"], tensors["state"],
            torch.ones(tensors["mask"].size(0), 5, device=device),
        )
    utility = output["utility"].cpu().numpy()
    mask = tensors["mask"].cpu().numpy()
    target = tensors["u_ce"].cpu().numpy()
    within, between, stds = [], [], []
    top1_choices = []
    for row in range(mask.shape[0]):
        valid = np.where(mask[row])[0]
        if valid.size == 0:
            continue
        values = utility[row, valid]
        within.append(float(values.var()))
        between.append(float(values.mean()))
        stds.append(float(values.std()))
        top1_choices.append(int(valid[int(np.argmax(values))]))
    within_var = float(np.mean(within))
    between_var = float(np.var(between))
    counts = np.bincount(top1_choices, minlength=5)
    probs = counts / max(1, counts.sum())
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
    # cross-fitted card-identity oracle
    dialogues = sorted(set(data["dialogues"]))
    rng = random.Random(66)
    rng.shuffle(dialogues)
    half = len(dialogues) // 2
    group_a, group_b = set(dialogues[:half]), set(dialogues[half:])
    card_utility = defaultdict(list)
    for index, dialogue in enumerate(data["dialogues"]):
        if dialogue not in group_a:
            continue
        for rank in np.where(mask[index])[0]:
            card_id = payload["records"][index]["candidate_ids"][rank]
            card_utility[card_id].append(float(target[index, rank]))
    mean_utility = {key: float(np.mean(v)) for key, v in card_utility.items()}
    regrets, random_regrets = [], []
    for index, dialogue in enumerate(data["dialogues"]):
        if dialogue not in group_b:
            continue
        valid = np.where(mask[index])[0]
        if valid.size == 0:
            continue
        best = max(0.0, float(target[index, valid].max()))
        ids = payload["records"][index]["candidate_ids"]
        scores = [mean_utility.get(ids[rank], 0.0) for rank in valid]
        chosen = int(np.argmax(scores))
        action_utility = (
            float(target[index, valid[chosen]]) if scores[chosen] > 0 else 0.0
        )
        regrets.append(best - action_utility)
        rand_choice = valid[rng.randrange(valid.size)]
        random_regrets.append(best - max(0.0, float(target[index, rand_choice])))
    report = {
        "queries": len(within),
        "within_query_variance": within_var,
        "between_query_variance": between_var,
        "rv_ratio": within_var / (between_var + 1e-9),
        "fraction_pred_std_lt_1e-4": float(np.mean(np.array(stds) < 1e-4)),
        "top1_choice_entropy": entropy,
        "crossfit_card_identity": {
            "group_a_dialogues": len(group_a),
            "group_b_dialogues": len(group_b),
            "mean_action_regret": float(np.mean(regrets)) if regrets else None,
            "mean_random_regret": float(np.mean(random_regrets)) if random_regrets else None,
            "indexed_cards": len(mean_utility),
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
