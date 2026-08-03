import torch
import torch.nn as nn
import torch.nn.functional as F


def rep_distill_loss(
    h_mem: torch.Tensor,
    h_gen: torch.Tensor,
    eps: float = 1e-8,
    detach_target: bool = True,
) -> torch.Tensor:
    """
    Representation distillation loss pulling reasoning layer representation h_gen
    towards early storage representation h_mem using Cosine distance.
    
    L_RepDist = 1 - cos(h_gen, sg[h_mem])
    
    Args:
        h_mem: Storage layer hidden state (B, D) or (B, N, D).
        h_gen: Reasoning layer hidden state (B, D) or (B, N, D).
        eps: Small epsilon for numerical stability during L2 normalization.
        detach_target: If True, applies stop-gradient operator (detach) to h_mem.
    """
    if h_mem.dim() == 3:
        h_mem = h_mem.mean(dim=1)
    if h_gen.dim() == 3:
        h_gen = h_gen.mean(dim=1)

    target_state = h_mem.detach() if detach_target else h_mem

    h_mem_norm = F.normalize(target_state, p=2, dim=-1, eps=eps)
    h_gen_norm = F.normalize(h_gen, p=2, dim=-1, eps=eps)

    # Cosine similarity per batch item
    cos_sim = (h_mem_norm * h_gen_norm).sum(dim=-1)
    loss = 1.0 - cos_sim.mean()
    return loss


class RepresentationDistillationLoss(nn.Module):
    """
    Module wrapper for Representation Distillation Loss.
    """
    def __init__(self, eps: float = 1e-8, detach_target: bool = True):
        super().__init__()
        self.eps = eps
        self.detach_target = detach_target

    def forward(self, h_mem: torch.Tensor, h_gen: torch.Tensor) -> torch.Tensor:
        return rep_distill_loss(h_mem, h_gen, eps=self.eps, detach_target=self.detach_target)
