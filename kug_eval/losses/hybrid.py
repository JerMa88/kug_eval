import torch
import torch.nn as nn
from kug_eval.losses.contrastive import contrastive_loss
from kug_eval.losses.rep_distill import rep_distill_loss


class HybridAlignmentLoss(nn.Module):
    """
    Convex combination of representation alignment (contrastive / cosine)
    and linear probe decodability loss.
    L_Hybrid = alpha * L_Align + (1 - alpha) * L_Probe
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        h_mem: torch.Tensor,
        h_gen: torch.Tensor,
        probe_loss_val: torch.Tensor,
    ) -> torch.Tensor:
        l_contra = contrastive_loss(h_mem, h_gen, temperature=self.temperature)
        return self.alpha * l_contra + (1.0 - self.alpha) * probe_loss_val


def hybrid_loss(
    h_mem: torch.Tensor,
    h_gen: torch.Tensor,
    probe_loss_val: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 0.07,
) -> torch.Tensor:
    l_contra = contrastive_loss(h_mem, h_gen, temperature=temperature)
    return alpha * l_contra + (1.0 - alpha) * probe_loss_val
