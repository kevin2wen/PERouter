from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ComP.per.batch import PERDatasetAdapter
from ComP.per.missing import MaskManifest
from ComP.per.types import MODALITIES
from ComP.train_per import (
    _apply_manifest,
    _build_student,
    _check_feature_paths,
    _dataset_and_splits,
    _make_loader,
)


def tensor_stats(values):
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    value = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def correlation(left, right):
    if len(left) < 2:
        return 0.0
    a = torch.tensor(left, dtype=torch.float64)
    b = torch.tensor(right, dtype=torch.float64)
    if float(a.std(unbiased=False)) < 1e-12 or float(b.std(unbiased=False)) < 1e-12:
        return 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask-rate", required=True, type=float)
    parser.add_argument("--gpu", type=int, default=0)
    args_cli = parser.parse_args()

    checkpoint = Path(args_cli.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "fair":
        raise ValueError("diagnostic requires a fair checkpoint")
    saved = dict(payload["args"])
    device = torch.device(f"cuda:{args_cli.gpu}" if torch.cuda.is_available() else "cpu")
    saved.update({"no_cuda": device.type == "cpu", "gpu": args_cli.gpu, "device": device})
    args = Namespace(**saved)
    _check_feature_paths(args)
    dataset, splits, _ = _dataset_and_splits(args)
    split = next(item for item in splits if int(item["fold"]) == 0)
    adim, tdim, vdim = dataset.get_featDim()
    test_adapter = PERDatasetAdapter(dataset, split["test"])
    test_loader = _make_loader(test_adapter, args, training=False)
    model = _build_student(args, device, adim, tdim, vdim, evidence_dim=1).eval()
    model.load_state_dict(payload["model"], strict=True)
    manifest_path = checkpoint.parent / "test_mask_manifest.csv"
    manifest = MaskManifest.load(str(manifest_path))

    rows = []
    final_errors = []
    missing_coord_masses = []
    by_modality = defaultdict(list)
    by_pattern = defaultdict(list)

    with torch.no_grad():
        for full_cpu in test_loader:
            batch = _apply_manifest(full_cpu, manifest, device)
            output = model(batch.features, batch.availability, evidence=None, return_intermediates=True)
            if output.coordinator_weights is None:
                raise RuntimeError("coordinator weights are required")
            final = output.logits.squeeze(-1)
            final_error = (final - batch.labels).abs()
            modality_errors = {
                name: (value.squeeze(-1) - batch.labels).abs()
                for name, value in output.modality_logits.items()
            }
            valid_utterance = batch.sequence_mask & batch.labels.ne(0)
            for b in range(batch.labels.size(0)):
                for t in range(batch.labels.size(1)):
                    if not bool(valid_utterance[b, t]):
                        continue
                    observed = batch.availability[b, t]
                    missing = ~observed
                    if not bool(missing.any()):
                        continue
                    weights = output.coordinator_weights[b, t]
                    missing_mass = float(weights.masked_fill(observed, 0).sum())
                    observed_mass = float(weights.masked_fill(missing, 0).sum())
                    pattern = "".join("1" if bool(v) else "0" for v in missing.tolist())
                    ferr = float(final_error[b, t])
                    final_errors.append(ferr)
                    missing_coord_masses.append(missing_mass)
                    by_pattern[pattern].append(
                        {
                            "final_error": ferr,
                            "missing_mass": missing_mass,
                            "observed_mass": observed_mass,
                        }
                    )
                    for m, modality in enumerate(MODALITIES):
                        if not bool(observed[m]):
                            continue
                        cerr = float(modality_errors[modality][b, t])
                        weight = float(weights[m])
                        helpful = cerr < ferr
                        underweighted = weight < (1.0 / 3.0)
                        helpful_but_underweighted = helpful and underweighted
                        row = {
                            "sample_id": batch.sample_ids[b],
                            "utterance_id": batch.utterance_ids[b][t],
                            "time_index": t,
                            "label": float(batch.labels[b, t]),
                            "prediction": float(final[b, t]),
                            "missing_pattern": pattern,
                            "missing_modality": modality,
                            "final_error": ferr,
                            "missing_modality_error": cerr,
                            "coordinator_weight_missing_modality": weight,
                            "coordinator_missing_mass": missing_mass,
                            "coordinator_observed_mass": observed_mass,
                            "missing_candidate_helpful": helpful,
                            "missing_weight_below_one_third": underweighted,
                            "helpful_but_underweighted": helpful_but_underweighted,
                        }
                        rows.append(row)
                        by_modality[modality].append(row)

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "utterance_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def group_summary(group):
        return {
            "count": len(group),
            "final_mae": tensor_stats([r["final_error"] for r in group])["mean"],
            "missing_candidate_mae": tensor_stats([r["missing_modality_error"] for r in group])["mean"],
            "mean_weight_on_missing_modality": tensor_stats([r["coordinator_weight_missing_modality"] for r in group])["mean"],
            "mean_missing_coordinator_mass": tensor_stats([r["coordinator_missing_mass"] for r in group])["mean"],
            "mean_observed_coordinator_mass": tensor_stats([r["coordinator_observed_mass"] for r in group])["mean"],
            "helpful_fraction": sum(bool(r["missing_candidate_helpful"]) for r in group) / max(1, len(group)),
            "helpful_but_underweighted_fraction": sum(bool(r["helpful_but_underweighted"]) for r in group) / max(1, len(group)),
            "corr_missing_mass_error": correlation(
                [r["coordinator_missing_mass"] for r in group],
                [r["final_error"] for r in group],
            ),
        }

    summary = {
        "checkpoint": str(checkpoint),
        "mask_rate": args_cli.mask_rate,
        "manifest": str(manifest_path),
        "device": str(device),
        "test_utterance_modality_rows": len(rows),
        "overall": group_summary(rows),
        "by_missing_modality": {
            modality: group_summary(group) for modality, group in sorted(by_modality.items())
        },
        "by_missing_pattern": {
            pattern: {
                "count": len(group),
                "final_mae": tensor_stats([r["final_error"] for r in group])["mean"],
                "mean_missing_coordinator_mass": tensor_stats([r["missing_mass"] for r in group])["mean"],
                "mean_observed_coordinator_mass": tensor_stats([r["observed_mass"] for r in group])["mean"],
                "corr_missing_mass_error": correlation(
                    [r["missing_mass"] for r in group],
                    [r["final_error"] for r in group],
                ),
            }
            for pattern, group in sorted(by_pattern.items())
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
