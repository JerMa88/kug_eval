import pytest
import torch
import torch.nn as nn
from kug_eval.models.hooks import RepresentationCache, get_layer_hook, register_hooks, get_model_layer_module
from kug_eval.models.tracing import compute_cosine_similarity_matrix, compute_linear_cka, LayerRoutingTracer


class MockLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class MockTransformerModel(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        out = x
        for layer in self.model.layers:
            out = layer(out)
        return out


class MockPeftWrapperModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)


def test_representation_cache_basic():
    cache = RepresentationCache()
    assert 0 not in cache
    cache.cache[0] = torch.ones(2, 64)
    assert 0 in cache
    assert cache[0].shape == (2, 64)
    assert cache.get(0) is not None
    assert cache.get(1) is None

    with pytest.raises(KeyError):
        _ = cache[99]

    cache.clear()
    assert 0 not in cache


def test_get_model_layer_module_architectures():
    model = MockTransformerModel(num_layers=4)
    layer0 = get_model_layer_module(model, 0)
    assert isinstance(layer0, MockLayer)

    peft_model = MockPeftWrapperModel(model)
    layer1 = get_model_layer_module(peft_model, 1)
    assert isinstance(layer1, MockLayer)

    with pytest.raises(IndexError):
        get_model_layer_module(model, 10)


def test_hook_registration_and_span_extraction():
    model = MockTransformerModel(num_layers=4, hidden_dim=32)
    cache = RepresentationCache()

    # Spans as (start, end)
    spans = [(1, 3), (2, 4)]
    handles = register_hooks(model, [0, 2], cache, entity_spans=spans, detach=False)

    dummy_input = torch.randn(2, 8, 32)
    _ = model(dummy_input)

    assert 0 in cache
    assert 2 in cache
    # Mean-pooled span outputs should have shape (batch_size=2, hidden_dim=32)
    assert cache[0].shape == (2, 32)
    assert cache[2].shape == (2, 32)

    for h in handles:
        h.remove()


def test_hook_out_of_bounds_span_handling():
    model = MockTransformerModel(num_layers=2, hidden_dim=16)
    cache = RepresentationCache()

    # Out of bounds span indices: (10, 20) for seq len 5
    spans = [(10, 20), (-5, 2)]
    handles = register_hooks(model, [0], cache, entity_spans=spans)

    dummy_input = torch.randn(2, 5, 16)
    _ = model(dummy_input)

    assert cache[0].shape == (2, 16)
    for h in handles:
        h.remove()


def test_tracing_cka_and_cosine():
    X = torch.randn(10, 32)
    # Perfectly identical matrix should have CKA ~ 1.0
    cka_self = compute_linear_cka(X, X)
    assert abs(cka_self - 1.0) < 1e-4

    Y = torch.randn(10, 32)
    sim_mat = compute_cosine_similarity_matrix(X, Y)
    assert sim_mat.shape == (10, 10)

    tracer = LayerRoutingTracer(num_layers=2)
    mem_cache = {0: X, 1: X}
    gen_cache = {0: Y, 1: Y}

    res = tracer.profile_layer_similarities(mem_cache, gen_cache)
    assert "cosine_similarity" in res
    assert "cka" in res
    assert (0, 0) in res["cosine_similarity"]
