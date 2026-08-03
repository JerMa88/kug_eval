import os
import json
import tempfile
import pytest
import torch
from kug_eval.data.schema import GeneralizationTaskItem, DataContractError
from kug_eval.data.dataset import (
    load_task_items_from_jsonl,
    GeneralizationDataset,
    PairedTaskDataset,
    get_dataloader,
)


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, truncation=True, max_length=512, padding="max_length", return_tensors="pt", add_special_tokens=True):
        words = text.split()
        # Map words to simple integer IDs
        ids = [hash(w) % 1000 + 2 for w in words]
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        attn_mask = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attn_mask = attn_mask + [0] * pad_len

        res = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn_mask], dtype=torch.long)
        }
        return res


def test_schema_valid_item():
    item = GeneralizationTaskItem(
        id="sample_001",
        document="The car wash is 50 meters away.",
        query="Should I walk or drive to the car wash?",
        target_entity="Drive",
        category="car_wash",
    )
    assert item.id == "sample_001"
    assert "50 meters" in item.get_memorization_prompt()
    assert "walk or drive" in item.get_generalization_prompt()


def test_schema_invalid_empty_fields():
    with pytest.raises((DataContractError, ValueError)):
        GeneralizationTaskItem(
            id="",
            document="Doc",
            query="Query",
            target_entity="Target",
        )

    with pytest.raises((DataContractError, ValueError)):
        GeneralizationTaskItem(
            id="001",
            document="   ",
            query="Query",
            target_entity="Target",
        )


def test_jsonl_loader_valid_and_edge_cases():
    sample_data = [
        {"id": "1", "document": "Doc 1", "query": "Q1", "target_entity": "Ans1", "category": "cat1"},
        {"id": "2", "document": "Doc 2", "query": "Q2", "target_entity": "Ans2"},
    ]
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as tmp:
        for item in sample_data:
            tmp.write(json.dumps(item) + "\n")
        tmp_path = tmp.name

    try:
        items = load_task_items_from_jsonl(tmp_path, strict=True)
        assert len(items) == 2
        assert items[0].category == "cat1"
        assert items[1].category == "general"

        dataset = GeneralizationDataset(tmp_path)
        assert len(dataset) == 2
        assert dataset[0].id == "1"
    finally:
        os.remove(tmp_path)


def test_jsonl_loader_malformed_strict_and_non_strict():
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as tmp:
        tmp.write(json.dumps({"id": "1", "document": "Doc 1", "query": "Q1", "target_entity": "Ans1"}) + "\n")
        tmp.write("INVALID JSON LINE\n")
        tmp.write(json.dumps({"id": "2", "document": "Doc 2", "query": "Q2", "target_entity": "Ans2"}) + "\n")
        tmp_path = tmp.name

    try:
        with pytest.raises(DataContractError):
            load_task_items_from_jsonl(tmp_path, strict=True)

        items = load_task_items_from_jsonl(tmp_path, strict=False)
        assert len(items) == 2
    finally:
        os.remove(tmp_path)


def test_paired_dataset_and_dataloader():
    tokenizer = DummyTokenizer()
    sample_data = [
        {"id": "1", "document": "Doc 1 about EntityA", "query": "Query for EntityA", "target_entity": "EntityA"}
    ]
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as tmp:
        for item in sample_data:
            tmp.write(json.dumps(item) + "\n")
        tmp_path = tmp.name

    try:
        paired_ds = PairedTaskDataset(tmp_path, tokenizer, max_length=64)
        assert len(paired_ds) == 1
        sample = paired_ds[0]
        assert "mem_input_ids" in sample
        assert "gen_input_ids" in sample
        assert sample["mem_span"].shape == (2,)

        loader = get_dataloader(tmp_path, tokenizer, batch_size=1, max_length=64)
        for batch in loader:
            assert batch["mem_input_ids"].shape == (1, 64)
            assert batch["gen_input_ids"].shape == (1, 64)
    finally:
        os.remove(tmp_path)
