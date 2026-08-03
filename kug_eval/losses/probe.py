import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbeLoss(nn.Module):
    """
    Linear Probe Distillation Loss.
    Projects middle-layer reasoning states h_gen through a trainable linear probe phi: R^d -> R^|V|
    or projects to target entity dimension before cross-entropy.
    """
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.probe = nn.Linear(hidden_dim, vocab_size)

    def forward(self, h_gen: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_gen: Reasoning state (B, D) or (B, N, D)
            target_ids: Token ID labels (B,) or (B, N)
        """
        if h_gen.dim() == 3 and target_ids.dim() == 1:
            h_gen = h_gen.mean(dim=1)
            
        logits = self.probe(h_gen)
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
        loss = F.cross_entropy(logits, target_ids, ignore_index=-100)
        return loss


def probe_loss(h_gen: torch.Tensor, target_ids: torch.Tensor, probe: nn.Module) -> torch.Tensor:
    logits = probe(h_gen)
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        target_ids = target_ids.view(-1)
    return F.cross_entropy(logits, target_ids, ignore_index=-100)
