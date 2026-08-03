import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from kug_eval.data.schema import GeneralizationTaskItem, DataContractError

logger = logging.getLogger(__name__)


def load_task_items_from_jsonl(jsonl_path: str, strict: bool = True) -> List[GeneralizationTaskItem]:
    """
    Loads and validates GeneralizationTaskItem instances from a .jsonl file.
    
    Args:
        jsonl_path: Absolute or relative path to the JSONL file.
        strict: If True, raises DataContractError on invalid items; if False, logs warning and skips.
    """
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                item = GeneralizationTaskItem(**data)
                items.append(item)
            except Exception as e:
                err_msg = f"Error parsing JSONL line {line_idx} in {jsonl_path}: {e}"
                if strict:
                    raise DataContractError(err_msg) from e
                else:
                    logger.warning(err_msg)
    return items


class GeneralizationDataset(Dataset):
    """
    Basic PyTorch dataset wrapping GeneralizationTaskItem objects.
    """
    def __init__(self, jsonl_path: str, strict: bool = True):
        self.jsonl_path = jsonl_path
        self.items: List[GeneralizationTaskItem] = load_task_items_from_jsonl(jsonl_path, strict=strict)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GeneralizationTaskItem:
        return self.items[idx]


class PairedTaskDataset(Dataset):
    """
    PyTorch dataset for dual-prompt alignment training (P_mem and P_gen).
    Finds exact target entity spans in both memorization and generalization tokenized sequences.
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512, strict: bool = True):
        self.items: List[GeneralizationTaskItem] = load_task_items_from_jsonl(jsonl_path, strict=strict)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def _find_entity_span(self, token_ids: List[int], entity_ids: List[int]) -> Tuple[int, int]:
        """
        Locates start and end indices of target entity tokens within sequence token_ids.
        Falls back to last N non-padding tokens if exact sub-sequence match fails.
        """
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        # Determine actual sequence end before padding
        end_idx = len(token_ids)
        for i in range(len(token_ids)):
            if token_ids[i] == pad_id:
                end_idx = i
                break

        # Search for exact sublist match
        len_sub = len(entity_ids)
        if len_sub > 0:
            for start in range(end_idx - len_sub, -1, -1):
                if token_ids[start : start + len_sub] == entity_ids:
                    return start, start + len_sub

        # Fallback: estimate span at the tail end of non-padding tokens
        start_idx = max(0, end_idx - max(1, len_sub))
        return start_idx, max(start_idx + 1, end_idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        
        p_mem_text = item.get_memorization_prompt()
        p_gen_text = item.get_generalization_prompt()

        mem_enc = self.tokenizer(
            p_mem_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        gen_enc = self.tokenizer(
            p_gen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        target_enc = self.tokenizer(item.target_entity, add_special_tokens=False)
        if isinstance(target_enc, dict):
            target_ids = target_enc.get("input_ids", [])
        elif hasattr(target_enc, "input_ids"):
            target_ids = target_enc.input_ids
        elif isinstance(target_enc, list):
            target_ids = target_enc
        else:
            target_ids = list(target_enc)
            
        if isinstance(target_ids, torch.Tensor):
            target_ids = target_ids.squeeze().tolist()
            if isinstance(target_ids, int):
                target_ids = [target_ids]

        mem_input_ids = mem_enc["input_ids"][0] if isinstance(mem_enc, dict) else mem_enc.input_ids[0]
        mem_attention_mask = mem_enc["attention_mask"][0] if isinstance(mem_enc, dict) else mem_enc.attention_mask[0]
        gen_input_ids = gen_enc["input_ids"][0] if isinstance(gen_enc, dict) else gen_enc.input_ids[0]
        gen_attention_mask = gen_enc["attention_mask"][0] if isinstance(gen_enc, dict) else gen_enc.attention_mask[0]

        mem_ids = mem_input_ids.tolist()
        gen_ids = gen_input_ids.tolist()

        mem_span = self._find_entity_span(mem_ids, target_ids)
        gen_span = self._find_entity_span(gen_ids, target_ids)

        max_entity_len = 32
        target_ids_padded = target_ids[:max_entity_len] + [-100] * max(0, max_entity_len - len(target_ids))

        return {
            "id": item.id,
            "category": item.category,
            "mem_input_ids": mem_input_ids,
            "mem_attention_mask": mem_attention_mask,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "mem_span": torch.tensor(mem_span, dtype=torch.long),
            "gen_span": torch.tensor(gen_span, dtype=torch.long),
            "target_ids": torch.tensor(target_ids_padded, dtype=torch.long),
        }


def get_dataloader(
    jsonl_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
    strict: bool = True,
) -> DataLoader:
    dataset = PairedTaskDataset(jsonl_path, tokenizer, max_length=max_length, strict=strict)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
