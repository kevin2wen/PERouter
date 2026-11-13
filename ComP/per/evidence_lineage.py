from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class EvidenceRelation:
    same_utterance: bool
    same_video: bool
    same_dialogue: bool
    same_speaker: Optional[bool]
    adjacent_utterance: bool
    cross_video: bool


def split_candidate_id(candidate_id: str) -> tuple[str, str]:
    if ":" not in str(candidate_id):
        raise ValueError(f"invalid candidate id: {candidate_id!r}")
    sample, utterance = str(candidate_id).split(":", 1)
    return sample, utterance


def _utterance_index(value: str) -> Optional[int]:
    match = re.search(r"_(\d+)$", str(value))
    return int(match.group(1)) if match else None


def classify_relation(
    target_sample: str,
    target_utterance: str,
    candidate_id: str,
    *,
    same_speaker: Optional[bool] = None,
) -> EvidenceRelation:
    candidate_sample, candidate_utterance = split_candidate_id(candidate_id)
    same_video = candidate_sample == str(target_sample)
    same_utterance = same_video and candidate_utterance == str(target_utterance)
    target_index = _utterance_index(target_utterance)
    candidate_index = _utterance_index(candidate_utterance)
    adjacent = (
        same_video
        and not same_utterance
        and target_index is not None
        and candidate_index is not None
        and abs(target_index - candidate_index) == 1
    )
    return EvidenceRelation(
        same_utterance=same_utterance,
        same_video=same_video,
        same_dialogue=same_video,
        same_speaker=same_speaker,
        adjacent_utterance=adjacent,
        cross_video=not same_video,
    )


def aggregate_relations(relations: Iterable[EvidenceRelation]) -> Dict[str, float | int]:
    values = list(relations)
    total = len(values)
    if total == 0:
        return {
            "candidates": 0, "same_utterance_rate": 0.0, "same_video_rate": 0.0,
            "same_dialogue_rate": 0.0, "same_speaker_rate": None,
            "adjacent_utterance_rate": 0.0, "cross_video_rate": 0.0,
        }
    return {
        "candidates": total,
        "same_utterance_rate": sum(value.same_utterance for value in values) / total,
        "same_video_rate": sum(value.same_video for value in values) / total,
        "same_dialogue_rate": sum(value.same_dialogue for value in values) / total,
        "same_speaker_rate": None,
        "adjacent_utterance_rate": sum(value.adjacent_utterance for value in values) / total,
        "cross_video_rate": sum(value.cross_video for value in values) / total,
    }
