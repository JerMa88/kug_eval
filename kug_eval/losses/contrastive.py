import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(
    h_mem: torch.Tensor,
    h_gen: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    InfoNCE Contrastive loss aligning h_gen with corresponding target h_mem
    while pushing distractor batch entities apart.
    
    Args:
        h_mem: Storage layer hidden state (B, D)
        h_gen: Reasoning layer hidden state (B, D)
        temperature: Softmax scaling temperature (default: 0.07)
    """
    if h_mem.dim() == 3:
        h_mem = h_mem.mean(dim=1)
    if h_gen.dim() == 3:
        h_gen = h_gen.mean(dim=1)

    batch_size = h_gen.size(0)
    if batch_size == 1:
        # For batch size of 1, InfoNCE degenerates to Cosine distance
        return 1.0 - F.cosine_similarity(h_gen, h_mem.detach(), dim=-1, eps=eps).mean()

    h_mem_norm = F.normalize(h_mem.detach(), p=2, dim=-1, eps=eps)
    h_gen_norm = F.normalize(h_gen, p=2, dim=-1, eps=eps)

    # Similarity matrix (B, B)
    sim_matrix = torch.matmul(h_gen_norm, h_mem_norm.t()) / max(temperature, 1e-5)
    labels = torch.arange(batch_size, device=h_gen.device)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


class ContrastiveLoss(nn.Module):
    """
    Module wrapper for InfoNCE Contrastive Loss.
    """
    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, h_mem: torch.Tensor, h_gen: torch.Tensor) -> torch.Tensor:
        return contrastive_loss(h_mem, h_gen, temperature=self.temperature, eps=self.eps)
