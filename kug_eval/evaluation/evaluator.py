"""
kug_eval.evaluation.evaluator
==============================
Multi-model evaluator engine supporting:
  - Local PyTorch / HuggingFace Transformers models (GPU KV-cache)
  - SOTA Frontier API models with VERIFIED real model IDs (confirmed via live API queries 2026-08-05)

Logging: Every API call (full payload + full raw response) is written to a JSONL
         log file for retrospective analysis. Nothing is ever omitted.
"""

import os
import time
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import threading
import torch

from kug_eval.data.schema import GeneralizationTaskItem
from kug_eval.evaluation.metrics import exact_match_score, contains_match_score

logger = logging.getLogger(__name__)

# ── Thread safety for providers that require rate-limit serialization ──────────
_claude_lock = threading.Lock()


# ── Confirmed live model IDs (queried 2026-08-05 via each provider's /models API) ─
# OpenAI:    gpt-5.6-sol            (confirmed via GET /v1/models)
# Google:    gemini-3.6-flash       (confirmed via GET /v1beta/models, id=models/gemini-3.6-flash)
# Anthropic: claude-fable-5         (confirmed via GET /v1/models)
# Fireworks: deepseek-v4-flash      accounts/fireworks/models/deepseek-v4-flash
#            kimi-k3                accounts/fireworks/models/kimi-k3
#            qwen3p7-plus           accounts/fireworks/models/qwen3p7-plus  (Qwen 3.8 Max)
#            glm-5p2                accounts/fireworks/models/glm-5p2       (GLM 5.2)
MODEL_ID_MAP: Dict[str, str] = {
    # OpenAI
    "gpt-5.6-sol":          "gpt-5.6-sol",
    # Google
    "gemini-3.6-flash":     "gemini-3.6-flash",
    # Anthropic
    "claude-fable-5":       "claude-fable-5",
    # Fireworks
    "deepseek-v4-flash":    "accounts/fireworks/models/deepseek-v4-flash",
    "deepseek-v4-pro":      "accounts/fireworks/models/deepseek-v4-pro",
    "kimi-k3":              "accounts/fireworks/models/kimi-k3",
    "qwen3.8-max":          "accounts/fireworks/models/qwen3p7-plus",
    "qwen3p7-plus":         "accounts/fireworks/models/qwen3p7-plus",
    "glm-5.2":              "accounts/fireworks/models/glm-5p2",
    "glm-5p2":              "accounts/fireworks/models/glm-5p2",
    "minimax-m3":           "accounts/fireworks/models/minimax-m3",
}

# Provider detection keywords → routing function
_OPENAI_KEYWORDS    = ("gpt",)
_GEMINI_KEYWORDS    = ("gemini",)
_CLAUDE_KEYWORDS    = ("claude", "anthropic", "fable", "opus", "sonnet", "haiku")
_FIREWORKS_KEYWORDS = ("deepseek", "kimi", "qwen", "glm", "minimax", "llama",
                       "fireworks", "accounts/fireworks")


def _detect_provider(model_name: str) -> str:
    """Infer API provider from model name string."""
    ml = model_name.lower()
    if any(k in ml for k in _OPENAI_KEYWORDS):
        return "openai"
    if any(k in ml for k in _GEMINI_KEYWORDS):
        return "gemini"
    if any(k in ml for k in _CLAUDE_KEYWORDS):
        return "claude"
    if any(k in ml for k in _FIREWORKS_KEYWORDS):
        return "fireworks"
    return "fireworks"  # default fallback


def _strip_wrapper_prose(text: str, target: str) -> str:
    """
    Post-processor for verbose reasoning model outputs (e.g. Claude, MiniMax).

    If the raw generation wraps the answer in conversational prose, this
    attempts to extract just the core entity/value. Strategy:
      1. If normalized target is a substring of the text, return it verbatim
         (enables contains_match_score to fire but strict EM will still fail
         unless the prediction equals target after normalization).
      2. Return the text unchanged; the caller scores both strict EM and contains.

    NOTE: We do NOT alter the prediction string used for strict EM. This
    function is ONLY called to annotate the response in logs for the
    LLM-as-judge retrospective pass.
    """
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# API Logger — writes every call (payload + raw response) to a JSONL file
# ══════════════════════════════════════════════════════════════════════════════

class APICallLogger:
    """
    Appends one JSON record per API call to a JSONL log file.

    Each record contains:
      timestamp     UTC ISO-8601 timestamp
      model         model ID string sent to provider
      provider      openai | gemini | claude | fireworks
      item_id       task item ID (if available)
      prompt_type   'mem' | 'gen'
      payload       exact dict sent to the API (full, untruncated)
      raw_response  exact parsed JSON response from the API (full, untruncated)
      extracted_text  text we pulled out of the response
      latency_ms    round-trip latency in milliseconds
      error         error message if request failed, else null
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: Dict[str, Any]) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Global logger instance; set by evaluate_dataset() before any evaluation starts
_api_logger: Optional[APICallLogger] = None


def set_api_logger(log_path: str) -> None:
    global _api_logger
    _api_logger = APICallLogger(log_path)
    logger.info(f"[APICallLogger] Full API call logs will be written to: {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Base Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class BaseEvaluator(ABC):
    """Abstract base evaluator for local and API models."""

    @abstractmethod
    def generate_answer(self, prompt: str, item_id: str = "", prompt_type: str = "") -> str:
        """Generates a text answer string for a given prompt."""
        pass

    def evaluate_item(self, item: GeneralizationTaskItem) -> Dict[str, Any]:
        """
        Evaluates a single task item across both:
          P_mem  → A_mem  (factual context retrieval, no answer given)
          P_gen  → A_gen  (applied generalization, no context given)

        Returns both strict EM and contains-match scores.
        """
        p_mem = item.get_memorization_prompt()
        p_gen = item.get_generalization_prompt()

        pred_mem = self.generate_answer(p_mem, item_id=item.id, prompt_type="mem")
        pred_gen = self.generate_answer(p_gen, item_id=item.id, prompt_type="gen")

        # PRIMARY metric: strict exact match
        acc_mem = exact_match_score(pred_mem, item.target_entity)
        acc_gen = exact_match_score(pred_gen, item.target_entity)

        # SECONDARY metric: contains match (for judge analysis)
        contains_mem = contains_match_score(pred_mem, item.target_entity)
        contains_gen = contains_match_score(pred_gen, item.target_entity)

        return {
            "id":             item.id,
            "category":       item.category,
            "target_entity":  item.target_entity,
            "pred_mem":       pred_mem,
            "pred_gen":       pred_gen,
            # Primary strict EM
            "acc_mem":        acc_mem,
            "acc_gen":        acc_gen,
            # Secondary contains match
            "contains_mem":   contains_mem,
            "contains_gen":   contains_gen,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Local Model Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class LocalKVModelEvaluator(BaseEvaluator):
    """Evaluator for local PyTorch / HuggingFace Transformers models."""

    def __init__(self, model: Any, tokenizer: Any, device: Optional[str] = None,
                 max_new_tokens: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def generate_answer(self, prompt: str, item_id: str = "", prompt_type: str = "") -> str:
        raw_inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = ({k: v.to(self.device) if hasattr(v, "to") else v
                   for k, v in raw_inputs.items()}
                  if isinstance(raw_inputs, dict) else raw_inputs.to(self.device))

        with torch.no_grad():
            if hasattr(self.model, "generate"):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                return self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
            else:
                return self._fallback_generate(inputs)

    def _fallback_generate(self, inputs) -> str:
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = self.model(input_ids)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == (self.tokenizer.eos_token_id or -1):
                    break
        return self.tokenizer.decode(input_ids[0][seq_len:], skip_special_tokens=True).strip()


LocalModelEvaluator = LocalKVModelEvaluator


# ══════════════════════════════════════════════════════════════════════════════
# API Model Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class APIModelEvaluator(BaseEvaluator):
    """
    Evaluator for SOTA Frontier API models.

    Verified model IDs (queried 2026-08-05):
      OpenAI:    gpt-5.6-sol
      Google:    gemini-3.6-flash
      Anthropic: claude-fable-5
      Fireworks: deepseek-v4-flash, kimi-k3, qwen3p7-plus (Qwen 3.8 Max), glm-5p2 (GLM 5.2)

    Every HTTP call is logged in full (payload + raw response) to the configured
    APICallLogger. Use set_api_logger() before evaluation to activate logging.
    """

    def __init__(self, model_name: str = "gpt-5.6-sol", api_key: Optional[str] = None,
                 mock_mode: bool = False):
        self.model_name = model_name
        self.api_key = api_key or ""
        self.mock_mode = mock_mode
        self.provider = _detect_provider(model_name)
        logger.info(f"[APIModelEvaluator] model={model_name!r} provider={self.provider!r} "
                    f"mock={mock_mode}")

    def generate_answer(self, prompt: str, item_id: str = "", prompt_type: str = "") -> str:
        if self.mock_mode:
            return self._mock_api_generation(prompt)

        dispatch = {
            "openai":    self._call_openai_api,
            "gemini":    self._call_gemini_api,
            "claude":    self._call_claude_api,
            "fireworks": self._call_fireworks_api,
        }
        fn = dispatch.get(self.provider, self._call_fireworks_api)
        return fn(prompt, item_id=item_id, prompt_type=prompt_type)

    # ── HTTP helper ───────────────────────────────────────────────────────────

    def _http_post_json(self, url: str, headers: Dict[str, str],
                        payload: Dict[str, Any],
                        item_id: str = "", prompt_type: str = "",
                        max_retries: int = 5) -> Optional[Dict[str, Any]]:
        """
        POST JSON to URL with exponential back-off retry on 429 / 5xx.
        Logs every attempt (including failures) to the global APICallLogger.
        """
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        for attempt in range(max_retries):
            t0 = time.monotonic()
            error_msg = None
            raw_response = None
            extracted_text = ""

            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    res_body = response.read().decode("utf-8")
                    raw_response = json.loads(res_body)
                    latency_ms = int((time.monotonic() - t0) * 1000)

                    # Log successful call
                    if _api_logger:
                        _api_logger.log({
                            "timestamp":    datetime.now(timezone.utc).isoformat(),
                            "model":        self.model_name,
                            "provider":     self.provider,
                            "item_id":      item_id,
                            "prompt_type":  prompt_type,
                            "url":          url,
                            "payload":      payload,
                            "raw_response": raw_response,
                            "latency_ms":   latency_ms,
                            "attempt":      attempt + 1,
                            "error":        None,
                        })
                    return raw_response

            except urllib.error.HTTPError as e:
                latency_ms = int((time.monotonic() - t0) * 1000)
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")
                    raw_response = json.loads(error_body)
                except Exception:
                    raw_response = {"raw_error_body": error_body}
                error_msg = f"HTTPError {e.code}: {e.reason}"

                if _api_logger:
                    _api_logger.log({
                        "timestamp":    datetime.now(timezone.utc).isoformat(),
                        "model":        self.model_name,
                        "provider":     self.provider,
                        "item_id":      item_id,
                        "prompt_type":  prompt_type,
                        "url":          url,
                        "payload":      payload,
                        "raw_response": raw_response,
                        "latency_ms":   latency_ms,
                        "attempt":      attempt + 1,
                        "error":        error_msg,
                    })

                if e.code in (400, 401, 403, 404):
                    logger.error(f"[{self.model_name}] Non-retryable HTTP {e.code}. "
                                 f"Check API key / model ID. body={error_body[:300]}")
                    return None
                if e.code == 429:
                    # Distinguish billing exhaustion from transient rate limiting.
                    err_code = raw_response.get("error", {}).get("code", "") if raw_response else ""
                    if err_code in ("credit_balance_exhausted", "insufficient_quota", "billing_hard_limit_reached"):
                        logger.error(
                            f"[{self.model_name}] FATAL: Account has no credits remaining "
                            f"(code={err_code}). Top up at platform.openai.com/billing. "
                            f"Aborting — no point retrying."
                        )
                        raise RuntimeError(f"OpenAI billing exhausted: {err_code}")
                    # True rate limit — use aggressive back-off.
                    sleep_time = 60 + (30 * attempt)
                    logger.warning(f"[{self.model_name}] HTTP 429 Rate Limit — "
                                   f"sleeping {sleep_time}s (attempt {attempt+1}/{max_retries}) ...")
                else:
                    sleep_time = 2 ** attempt
                    logger.warning(f"[{self.model_name}] HTTP {e.code} attempt {attempt+1}/{max_retries}. "
                                   f"Sleeping {sleep_time}s ...")
                if attempt < max_retries - 1:
                    time.sleep(sleep_time)

            except Exception as e:
                latency_ms = int((time.monotonic() - t0) * 1000)
                error_msg = str(e)
                if _api_logger:
                    _api_logger.log({
                        "timestamp":    datetime.now(timezone.utc).isoformat(),
                        "model":        self.model_name,
                        "provider":     self.provider,
                        "item_id":      item_id,
                        "prompt_type":  prompt_type,
                        "url":          url,
                        "payload":      payload,
                        "raw_response": None,
                        "latency_ms":   latency_ms,
                        "attempt":      attempt + 1,
                        "error":        error_msg,
                    })
                logger.warning(f"[{self.model_name}] Request failed attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _call_openai_api(self, prompt: str, item_id: str = "",
                         prompt_type: str = "") -> str:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.error("OPENAI_API_KEY not set.")
            return ""

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # Use the verified real model ID from MODEL_ID_MAP, fallback to name
        real_model = MODEL_ID_MAP.get(self.model_name.lower(), self.model_name)

        # gpt-5.x and o-series are reasoning models: they only accept temperature=1 (default).
        # Omit temperature entirely for these models so the API uses its default.
        # Non-reasoning models (gpt-4o, gpt-4.x) accept temperature=0 for determinism.
        _reasoning_prefixes = ("o1", "o3", "o4", "gpt-5")
        is_reasoning = any(real_model.startswith(p) for p in _reasoning_prefixes)

        payload: Dict[str, Any] = {
            "model":                 real_model,
            "messages":              [{"role": "user", "content": prompt}],
            "max_completion_tokens": 300,
        }
        if not is_reasoning:
            payload["temperature"] = 0.0

        res = self._http_post_json(url, headers, payload, item_id=item_id,
                                   prompt_type=prompt_type)
        # gpt-5.6-sol rate limit is ~50 RPM. With 1 thread and 1.5s sleep we stay
        # at ~40 RPM — safely under budget. Do NOT use multiple threads for gpt-5.x.
        time.sleep(1.5)
        if res and "choices" in res and res["choices"]:
            return res["choices"][0]["message"]["content"].strip()
        return ""

    # ── Google Gemini ─────────────────────────────────────────────────────────

    def _call_gemini_api(self, prompt: str, item_id: str = "",
                         prompt_type: str = "") -> str:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.error("GEMINI_API_KEY not set.")
            return ""

        # Confirmed API endpoint for gemini-3.6-flash
        model_id = MODEL_ID_MAP.get(self.model_name.lower(), self.model_name)
        # Strip "models/" prefix if present for URL construction
        model_id_clean = model_id.replace("models/", "")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_id_clean}:generateContent?key={api_key}")
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 150},
        }

        res = self._http_post_json(url, headers, payload, item_id=item_id,
                                   prompt_type=prompt_type)
        if res and "candidates" in res and res["candidates"]:
            parts = res["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
        return ""

    # ── Anthropic Claude ──────────────────────────────────────────────────────

    def _call_claude_api(self, prompt: str, item_id: str = "",
                         prompt_type: str = "") -> str:
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set.")
            return ""

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        }

        model_id = MODEL_ID_MAP.get(self.model_name.lower(), self.model_name)
        payload = {
            "model":      model_id,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }

        with _claude_lock:
            res = self._http_post_json(url, headers, payload, item_id=item_id,
                                       prompt_type=prompt_type)
            time.sleep(0.3)

        if res and "content" in res and isinstance(res["content"], list):
            pieces = [
                b.get("text", "")
                for b in res["content"]
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            return "".join(pieces).strip()
        return ""

    # ── Fireworks AI ──────────────────────────────────────────────────────────

    def _call_fireworks_api(self, prompt: str, item_id: str = "",
                            prompt_type: str = "") -> str:
        api_key = (self.api_key or os.getenv("FIREWORKS_AI_API_KEY", "")
                   or os.getenv("FIREWORKS_API_KEY", ""))
        if not api_key:
            logger.error("FIREWORKS_AI_API_KEY not set.")
            return ""

        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # Resolve to confirmed Fireworks model path
        fw_model = MODEL_ID_MAP.get(self.model_name.lower(), self.model_name)
        # If user passed a Fireworks path directly, trust it
        if not fw_model.startswith("accounts/fireworks"):
            fw_model = f"accounts/fireworks/models/{fw_model}"

        payload = {
            "model":       fw_model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  150,
            "temperature": 0.0,
        }

        res = self._http_post_json(url, headers, payload, item_id=item_id,
                                   prompt_type=prompt_type)
        if res and "choices" in res and res["choices"]:
            msg = res["choices"][0].get("message", {})
            text = msg.get("content") or msg.get("reasoning_content") or ""
            return text.strip() if isinstance(text, str) else ""
        return ""

    # ── Mock mode ─────────────────────────────────────────────────────────────

    def _mock_api_generation(self, prompt: str) -> str:
        """Deterministic mock for offline unit tests."""
        pl = prompt.lower()
        if "drive" in pl:
            return "Drive"
        if "walk" in pl:
            return "Walk"
        if "university college london" in pl:
            return "University College London"
        return "MockEntity"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Evaluation Runner
# ══════════════════════════════════════════════════════════════════════════════

from concurrent.futures import ThreadPoolExecutor, as_completed


def evaluate_dataset(
    evaluator: BaseEvaluator,
    task_items: List[GeneralizationTaskItem],
    max_workers: int = 5,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates a full collection of task items and aggregates:
      - Strict EM accuracy (primary)
      - Contains-match accuracy (secondary, for judge analysis)
      - KUG ratio per category and overall

    Parameters
    ----------
    evaluator   : BaseEvaluator instance (local or API)
    task_items  : List of GeneralizationTaskItem
    max_workers : Thread pool size for concurrent API calls
    log_path    : If set, activates full API call logging to this JSONL path
    """
    if log_path:
        set_api_logger(log_path)

    results = []
    cat_mem, cat_gen = {}, {}
    cat_contains_mem, cat_contains_gen = {}, {}

    total_items = len(task_items)
    workers = max_workers if isinstance(evaluator, APIModelEvaluator) else 1

    if workers > 1:
        logger.info(f"Parallelizing API evaluations across {workers} worker threads...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(evaluator.evaluate_item, item): item
                for item in task_items
            }
            completed = 0
            for future in as_completed(future_to_item):
                res = future.result()
                results.append(res)
                cat = res["category"] or "general"
                cat_mem.setdefault(cat, []).append(res["acc_mem"])
                cat_gen.setdefault(cat, []).append(res["acc_gen"])
                cat_contains_mem.setdefault(cat, []).append(res["contains_mem"])
                cat_contains_gen.setdefault(cat, []).append(res["contains_gen"])
                completed += 1
                if completed % 50 == 0 or completed == total_items:
                    logger.info(f"Progress: {completed}/{total_items} "
                                f"({completed/total_items*100:.1f}%)")
    else:
        for idx, item in enumerate(task_items):
            res = evaluator.evaluate_item(item)
            results.append(res)
            cat = item.category or "general"
            cat_mem.setdefault(cat, []).append(res["acc_mem"])
            cat_gen.setdefault(cat, []).append(res["acc_gen"])
            cat_contains_mem.setdefault(cat, []).append(res["contains_mem"])
            cat_contains_gen.setdefault(cat, []).append(res["contains_gen"])
            if (idx + 1) % 50 == 0 or (idx + 1) == total_items:
                logger.info(f"Progress: {idx+1}/{total_items} "
                            f"({(idx+1)/total_items*100:.1f}%)")

    all_mem      = [r["acc_mem"]      for r in results]
    all_gen      = [r["acc_gen"]      for r in results]
    all_cmem     = [r["contains_mem"] for r in results]
    all_cgen     = [r["contains_gen"] for r in results]

    def _avg(lst):
        return float(sum(lst) / len(lst)) if lst else 0.0

    overall_a_mem     = _avg(all_mem)
    overall_a_gen     = _avg(all_gen)
    overall_cmem      = _avg(all_cmem)
    overall_cgen      = _avg(all_cgen)
    kug_ratio         = overall_a_mem / max(overall_a_gen, 1e-5)
    kug_ratio_contain = overall_cmem / max(overall_cgen, 1e-5)

    category_summary = {}
    for cat in cat_mem:
        avg_mem  = _avg(cat_mem[cat])
        avg_gen  = _avg(cat_gen[cat])
        avg_cmem = _avg(cat_contains_mem.get(cat, [0.0]))
        avg_cgen = _avg(cat_contains_gen.get(cat, [0.0]))
        category_summary[cat] = {
            "a_mem":         avg_mem,
            "a_gen":         avg_gen,
            "kug_ratio":     float(avg_mem / max(avg_gen, 1e-5)),
            "contains_mem":  avg_cmem,
            "contains_gen":  avg_cgen,
            "count":         len(cat_mem[cat]),
        }

    return {
        # Primary strict-EM metrics
        "overall_a_mem":         overall_a_mem,
        "overall_a_gen":         overall_a_gen,
        "kug_ratio":             float(kug_ratio),
        # Secondary contains-match metrics (for judge retrospective)
        "overall_contains_mem":  overall_cmem,
        "overall_contains_gen":  overall_cgen,
        "kug_ratio_contains":    float(kug_ratio_contain),
        "total_count":           len(results),
        "category_summary":      category_summary,
        "item_results":          results,
    }
