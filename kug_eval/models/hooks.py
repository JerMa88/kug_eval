from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn


class RepresentationCache:
    """
    Thread-safe container storing layer hidden states extracted during forward passes.
    """
    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.cache.clear()

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.cache.get(layer_idx)

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self.cache:
            raise KeyError(f"Layer index {layer_idx} not present in representation cache.")
        return self.cache[layer_idx]

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self.cache


def get_layer_hook(
    layer_idx: int,
    cache: RepresentationCache,
    entity_spans: Optional[List[Union[Tuple[int, int], List[int]]]] = None,
    detach: bool = False,
):
    """
    Returns a forward hook closure that extracts intermediate hidden states.
    
    Args:
        layer_idx: Layer index identifier.
        cache: RepresentationCache instance.
        entity_spans: List of token spans [(start, end)] or token indices for each batch item.
        detach: If True, detaches the tensor from the autograd computation graph.
    """
    def hook(module: nn.Module, inputs: Tuple[Any, ...], outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # Ensure hidden_states has shape (batch_size, sequence_length, hidden_dim)
        if hidden_states.dim() == 2:
            if entity_spans is not None:
                batch_size = len(entity_spans)
                seq_len = hidden_states.size(0) // batch_size
                hidden_states = hidden_states.view(batch_size, seq_len, -1)
            else:
                hidden_states = hidden_states.unsqueeze(0)

        tensor_to_store = hidden_states.detach() if detach else hidden_states

        if entity_spans is None:
            cache.cache[layer_idx] = tensor_to_store
        else:
            batch_size = hidden_states.size(0)
            entity_reps = []
            for i in range(batch_size):
                span = entity_spans[i]
                if isinstance(span, (tuple, list)) and len(span) == 2 and isinstance(span[0], int) and isinstance(span[1], int):
                    start, end = span[0], span[1]
                    if start >= end or start < 0 or end > hidden_states.size(1):
                        span_states = hidden_states[i, max(0, min(start, hidden_states.size(1)-1)):max(1, min(end, hidden_states.size(1))), :]
                    else:
                        span_states = hidden_states[i, start:end, :]
                else:
                    span_indices = torch.tensor(span, dtype=torch.long, device=hidden_states.device)
                    span_states = hidden_states[i, span_indices, :]
                
                mean_pooled = span_states.mean(dim=0)
                entity_reps.append(mean_pooled)

            stacked_reps = torch.stack(entity_reps)
            cache.cache[layer_idx] = stacked_reps.detach() if detach else stacked_reps

        return outputs

    return hook


def get_model_layer_module(model: nn.Module, layer_idx: int) -> nn.Module:
    """
    Robustly resolves the PyTorch nn.Module corresponding to layer_idx across standard
    and custom transformer architectures (Llama, Qwen, DeepSeek, Mistral, Gemma, PEFT LoRA wrappers).
    """
    base = model
    if hasattr(base, "base_model"):
        base = getattr(base, "base_model")
        if hasattr(base, "model"):
            base = getattr(base, "model")

    # Standard Llama / Qwen / Mistral / Gemma
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        layers = base.model.layers
    elif hasattr(base, "layers"):
        layers = base.layers
    # GPT-NeoX / Pythia
    elif hasattr(base, "gpt_neox") and hasattr(base.gpt_neox, "layers"):
        layers = base.gpt_neox.layers
    # OPT models
    elif hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        layers = base.decoder.layers
    # Custom modular split (e.g. HRM-Text L_module and H_module)
    elif hasattr(base, "model") and hasattr(base.model, "L_module"):
        if layer_idx < 16:
            return base.model.L_module.layers[layer_idx]
        else:
            return base.model.H_module.layers[layer_idx - 16]
    elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        layers = base.transformer.h
    else:
        raise ValueError(
            f"Unsupported model architecture for layer hook registration. Module structure: {dir(base)}"
        )

    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"Layer index {layer_idx} out of range for model with {len(layers)} layers.")

    return layers[layer_idx]


def register_hooks(
    model: nn.Module,
    layer_indices: List[int],
    cache: RepresentationCache,
    entity_spans: Optional[List[Union[Tuple[int, int], List[int]]]] = None,
    detach: bool = False,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Registers forward hooks on specified layer indices.
    """
    handles = []
    for idx in layer_indices:
        layer_module = get_model_layer_module(model, idx)
        handle = layer_module.register_forward_hook(get_layer_hook(idx, cache, entity_spans, detach=detach))
        handles.append(handle)
    return handles
