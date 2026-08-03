import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def compute_cosine_similarity_matrix(h_mem: torch.Tensor, h_gen: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Cosine Similarity between memorization hidden states (B, D)
    and generalization hidden states (B, D).
    Returns (B, B) similarity matrix.
    """
    h_mem_norm = F.normalize(h_mem, p=2, dim=-1)
    h_gen_norm = F.normalize(h_gen, p=2, dim=-1)
    return torch.matmul(h_gen_norm, h_mem_norm.transpose(-1, -2))


def compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between two representation matrices X (N, D1) and Y (N, D2).
    CKA measures structural representation similarity invariant to orthogonal rotation and isotropic scaling.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Gram matrices
    K = torch.matmul(X, X.t())
    L = torch.matmul(Y, Y.t())
    
    hsic_KL = (K * L).sum()
    hsic_KK = (K * K).sum()
    hsic_LL = (L * L).sum()
    
    denom = torch.sqrt(hsic_KK * hsic_LL)
    if denom == 0:
        return 0.0
    return (hsic_KL / denom).item()


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Convenience alias for linear CKA."""
    return compute_linear_cka(X, Y)


class LayerRoutingTracer:
    """
    Diagnostic tool to profile layer-wise Signal-to-Noise Ratio (SNR) and layer routing similarities
    between storage layers l_s and reasoning layers l_t.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers

    def profile_layer_similarities(
        self,
        mem_cache: Dict[int, torch.Tensor],
        gen_cache: Dict[int, torch.Tensor],
    ) -> Dict[str, Dict[Tuple[int, int], float]]:
        """
        Profiles cross-layer cosine similarity and CKA between all (l_src, l_tgt) pairs.
        """
        cosine_results = {}
        cka_results = {}
        
        for l_src, h_mem in mem_cache.items():
            for l_tgt, h_gen in gen_cache.items():
                # Ensure 2D (B, D)
                if h_mem.dim() == 3:
                    h_mem_2d = h_mem.mean(dim=1)
                else:
                    h_mem_2d = h_mem

                if h_gen.dim() == 3:
                    h_gen_2d = h_gen.mean(dim=1)
                else:
                    h_gen_2d = h_gen

                # Diagonal mean cosine similarity
                cos_sim = F.cosine_similarity(h_mem_2d, h_gen_2d, dim=-1).mean().item()
                cka_val = compute_linear_cka(h_mem_2d, h_gen_2d)

                cosine_results[(l_src, l_tgt)] = cos_sim
                cka_results[(l_src, l_tgt)] = cka_val

        return {
            "cosine_similarity": cosine_results,
            "cka": cka_results,
        }
