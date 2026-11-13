import unittest

import torch
import torch.nn as nn

from ComP.per.evidence_adapter import (
    adapter_parameter_count,
    configure_adapter_training,
)


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.evidence_encoder = nn.Linear(3, 4)


class EvidenceAdapterTest(unittest.TestCase):
    def test_only_evidence_encoder_is_trainable(self):
        model = ToyStudent()
        configure_adapter_training(model)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.evidence_encoder.parameters()))
        self.assertEqual(adapter_parameter_count(model), 16)

    def test_student_stays_in_eval_and_adapter_in_train_mode(self):
        model = ToyStudent()
        configure_adapter_training(model)
        self.assertFalse(model.training)
        self.assertTrue(model.evidence_encoder.training)
