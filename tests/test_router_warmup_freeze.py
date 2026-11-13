from __future__ import annotations

import unittest

from ComP.train_per import _configure_route_training
from tests.helpers import tiny_args, tiny_model


class RouterWarmupFreezeTest(unittest.TestCase):
    def test_warmup_freezes_candidate_representation_not_router(self):
        args = tiny_args()
        args.freeze_backbone_epochs = 0
        args.router_warmup_epochs = 2
        args.freeze_candidate_during_warmup = True
        args.stage_epoch = 150
        model = tiny_model(use_router=True)

        active = _configure_route_training(model, args, epoch=150, first_stage=False)
        self.assertTrue(active)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.evidence_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.observed_projections.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.router.parameters()))

        inactive = _configure_route_training(model, args, epoch=152, first_stage=False)
        self.assertFalse(inactive)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.evidence_encoder.parameters()))

    def test_default_warmup_is_disabled(self):
        args = tiny_args()
        args.freeze_backbone_epochs = 0
        args.router_warmup_epochs = 0
        args.freeze_candidate_during_warmup = False
        args.stage_epoch = 150
        model = tiny_model(use_router=True)
        active = _configure_route_training(model, args, epoch=150, first_stage=False)
        self.assertFalse(active)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
