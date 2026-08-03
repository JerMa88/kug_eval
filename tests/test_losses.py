import pytest
import torch
import torch.nn as nn
from kug_eval.losses import (
    RepresentationDistillationLoss,
    rep_distill_loss,
    ContrastiveLoss,
    contrastive_loss,
    LinearProbeLoss,
    probe_loss,
    HybridAlignmentLoss,
    hybrid_loss,
)


def test_rep_distill_loss_identities():
    # Identical vectors should yield loss ~ 0.0
    h_mem = torch.randn(4, 32)
    loss_identical = rep_distill_loss(h_mem, h_mem.clone())
    assert abs(loss_identical.item()) < 1e-5

    # Opposite vectors should yield loss ~ 2.0
    loss_opposite = rep_distill_loss(h_mem, -h_mem)
    assert abs(loss_opposite.item() - 2.0) < 1e-4

    # Module wrapper
    mod = RepresentationDistillationLoss()
    res = mod(h_mem, h_mem)
    assert abs(res.item()) < 1e-5


def test_rep_distill_stop_gradient():
    h_mem = torch.randn(2, 16, requires_grad=True)
    h_gen = torch.randn(2, 16, requires_grad=True)

    loss = rep_distill_loss(h_mem, h_gen, detach_target=True)
    loss.backward()

    # h_mem should NOT receive gradients because detach_target=True
    assert h_mem.grad is None or torch.all(h_mem.grad == 0)
    # h_gen SHOULD receive gradients
    assert h_gen.grad is not None and not torch.all(h_gen.grad == 0)


def test_contrastive_loss_batch_sizes():
    # Batch size > 1
    h_mem = torch.randn(4, 32)
    h_gen = torch.randn(4, 32)
    loss_multi = contrastive_loss(h_mem, h_gen, temperature=0.1)
    assert loss_multi.item() > 0.0

    # Batch size = 1 fallback
    h_mem_single = torch.randn(1, 32)
    h_gen_single = torch.randn(1, 32)
    loss_single = contrastive_loss(h_mem_single, h_gen_single)
    assert 0.0 <= loss_single.item() <= 2.0


def test_linear_probe_loss_backward():
    probe_mod = LinearProbeLoss(hidden_dim=16, vocab_size=100)
    h_gen = torch.randn(4, 16, requires_grad=True)
    target_ids = torch.tensor([5, 12, 45, 99], dtype=torch.long)

    loss = probe_mod(h_gen, target_ids)
    assert loss.item() > 0.0

    loss.backward()
    assert h_gen.grad is not None
    assert probe_mod.probe.weight.grad is not None


def test_hybrid_loss():
    h_mem = torch.randn(4, 32)
    h_gen = torch.randn(4, 32)
    probe_val = torch.tensor(1.5, requires_grad=True)

    hyb_mod = HybridAlignmentLoss(alpha=0.6)
    res = hyb_mod(h_mem, h_gen, probe_val)
    assert res.item() > 0.0
