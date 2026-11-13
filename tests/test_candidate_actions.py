import unittest

import torch

from ComP.per.candidate_actions import (
    CandidateAction,
    action_candidates,
    make_candidate_cache_key,
    select_best_actions,
    validate_train_only_split,
)


class CandidateActionsTest(unittest.TestCase):
    def test_skip_and_rank_actions(self):
        actions = action_candidates(1, 3)
        self.assertEqual(actions[0], CandidateAction(False, -1, 1))
        self.assertEqual([a.rank for a in actions[1:]], [0, 1, 2])

    def test_best_action_and_threshold(self):
        utility = torch.tensor([[[[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
        skip = torch.tensor([[[0.15, 0.0, 0.0]]])
        active = torch.tensor([[[True, False, False]]])
        result = select_best_actions(utility, skip, active, 0.0)
        self.assertEqual(result["best_rank"].tolist(), [[[2, 0, 0]]])
        self.assertTrue(result["use_evidence"][0, 0, 0])
        no_help = select_best_actions(utility, skip, active, 0.2)
        self.assertFalse(no_help["use_evidence"][0, 0, 0])
        self.assertEqual(int(no_help["selected_rank"][0, 0, 0]), -1)

    def test_inactive_positions_force_skip(self):
        utility = torch.ones(1, 1, 3, 2)
        skip = torch.zeros(1, 1, 3)
        active = torch.tensor([[[False, False, False]]])
        result = select_best_actions(utility, skip, active, 0.0)
        self.assertFalse(result["use_evidence"].any())
        self.assertTrue((result["selected_rank"] == -1).all())

    def test_cache_key_changes_with_inputs(self):
        base = dict(
            dataset_version="v1", train_split_hash="a", student_checkpoint_hash="b",
            adapter_checkpoint_hash=None, evidence_source="feature",
            evidence_source_hash="c", evidence_encoder_hash="d",
            mask_manifest_hash="e", top_k=5, gamma=0.5,
            evidence_mode="feature", sampler_version="categorical",
        )
        first = make_candidate_cache_key(**base)
        changed = dict(base)
        changed["top_k"] = 10
        self.assertNotEqual(first, make_candidate_cache_key(**changed))

    def test_cache_key_changes_with_exclusion_policy(self):
        base = dict(
            dataset_version="v1", train_split_hash="a", student_checkpoint_hash="b",
            adapter_checkpoint_hash=None, evidence_source="feature",
            evidence_source_hash="c", evidence_encoder_hash="d",
            mask_manifest_hash="e", top_k=5, gamma=0.5,
            evidence_mode="feature", sampler_version="categorical",
            min_helpful_gain=0.0, exclusion_policy="utterance-only",
        )
        other = dict(base)
        other["exclusion_policy"] = "utterance+video+dialogue"
        self.assertNotEqual(make_candidate_cache_key(**base), make_candidate_cache_key(**other))

    def test_cache_generation_is_train_only(self):
        validate_train_only_split("train")
        with self.assertRaises(PermissionError):
            validate_train_only_split("validation")
