from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans

from ComP.per.types import MODALITIES

from .schema import KnowledgeCardV4


class CrossModalPrototypeBuilder:
    """K2-U: unsupervised train-only observed-to-target prototypes.

    Uses no labels and never reads the current query's missing target content.
    For each target modality, observed modalities of train utterances are
    clustered; each cluster stores the mean target representation as a
    prototype. Cards expose only aggregate statistics and provenance.
    """

    def __init__(self, n_clusters: int = 8, seed: int = 66) -> None:
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self._cards: List[KnowledgeCardV4] = []
        self._prototype_vectors: Dict[str, Dict[str, torch.Tensor]] = {}

    def fit(self, loader) -> None:
        per_modality = {name: [] for name in MODALITIES}
        sample_ids = []
        utterance_ids = []
        for batch in loader:
            for b in range(batch.sequence_mask.size(0)):
                for t in range(batch.sequence_mask.size(1)):
                    if not bool(batch.sequence_mask[b, t]):
                        continue
                    for name in MODALITIES:
                        per_modality[name].append(
                            batch.features[name][b, t].detach().cpu().numpy()
                        )
                    sample_ids.append(str(batch.sample_ids[b]))
                    utterance_ids.append(str(batch.utterance_ids[b][t]))
        if not sample_ids:
            raise ValueError("empty train loader")
        arrays = {
            name: np.stack(values).astype(np.float32)
            for name, values in per_modality.items()
        }
        for target in MODALITIES:
            observed = [name for name in MODALITIES if name != target]
            features = np.concatenate(
                [arrays[name] for name in observed], axis=1
            )
            clusters = KMeans(
                n_clusters=self.n_clusters, random_state=self.seed, n_init=10
            ).fit(features)
            labels = clusters.labels_
            vectors = {}
            for cluster in range(self.n_clusters):
                indices = np.where(labels == cluster)[0]
                if indices.size == 0:
                    continue
                target_mean = arrays[target][indices].mean(axis=0)
                source_samples = sorted({sample_ids[i] for i in indices})
                card_id = hashlib.sha256(
                    f"k2u|{target}|{cluster}|{self.seed}".encode()
                ).hexdigest()[:16]
                claim = (
                    f"Train-only prototype cluster {cluster} of observed "
                    f"{'+'.join(observed)} co-occurs with a {target} "
                    f"representation (norm {float(np.linalg.norm(target_mean)):.2f}); "
                    f"support={indices.size}"
                )
                card = KnowledgeCardV4(
                    card_id=card_id,
                    claim=claim,
                    source="train_only_cross_modal_prototype",
                    source_type="K2-U",
                    source_version="kb_v4",
                    source_confidence=min(1.0, indices.size / 50.0),
                    observed_modalities=tuple(observed),
                    handled_missing_modalities=(target,),
                    trigger_predicates=(
                        f"nearest observed cluster == {cluster}",
                        f"observed modalities = {'+'.join(observed)}",
                    ),
                    negative_conditions=(
                        f"query sample appears in cluster source videos",
                    ),
                    evidence_relation="complement",
                    expected_effect_direction="support_target_representation",
                    supporting_context=tuple(source_samples[:10]),
                    provenance={
                        "label_used": False,
                        "target_raw_used_for_current_query": False,
                        "support": int(indices.size),
                        "cluster_center_norm": float(
                            np.linalg.norm(clusters.cluster_centers_[cluster])
                        ),
                    },
                )
                self._cards.append(card)
                vectors[card.card_id] = torch.from_numpy(target_mean)
            self._prototype_vectors[target] = vectors

    @property
    def cards(self) -> List[KnowledgeCardV4]:
        return list(self._cards)

    def prototype_vectors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            target: dict(vectors)
            for target, vectors in self._prototype_vectors.items()
        }
