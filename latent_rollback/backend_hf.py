"""
HuggingFace Transformers backend for RSCE.

Drop-in replacement for backend_mlx.py — same public interface:
  load_model(hf_id)                -> HFModelWrapper
  extract_context_state(...)       -> (torch.Tensor, int)
  generate_with_context_injection(...)  -> (str, int)
  generate_baseline_qa(...)        -> (str, int)

Injection mechanism:
  register_forward_hook on model.model.layers[L].
  The hook adds scale * context_vector to every token position of the
  residual stream output at that layer.

Extraction mechanism:
  Single forward pass with output_hidden_states=True.
  hidden_states[L+1] (index 0 = embedding) mean-pooled across all positions.

Compatible architectures (all use the same decoder layer output format):
  LLaMA-3, Qwen2.5, Mistral, DeepSeek-R1-Distill
"""

from __future__ import annotations

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_EPSILON = 1e-8
QA_STOP_STRINGS: tuple[str, ...] = ("\n", "\nQuestion:", "\nFacts:", "\nContext:")


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class HFModelWrapper:
    def __init__(self, model, tokenizer, hf_id: str):
        self._model = model
        self._tokenizer = tokenizer
        self.hf_id = hf_id
        self.n_layers: int = model.config.num_hidden_layers
        self.d_model: int = model.config.hidden_size

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    @property
    def model(self):
        return self._model

    @property
    def layers(self):
        # Works for LLaMA, Qwen, Mistral, DeepSeek (all store as model.model.layers)
        return self._model.model.layers


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(hf_id: str, device: str = "cuda") -> HFModelWrapper:
    print(f"Loading {hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    wrapper = HFModelWrapper(model, tokenizer, hf_id)
    print(f"Loaded. Layers: {wrapper.n_layers}  d_model: {wrapper.d_model}")
    return wrapper


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_context_state(
    wrapper: HFModelWrapper,
    context_text: str,
    layer: int,
    pool: str = "mean",
) -> tuple[torch.Tensor, int]:
    """
    Extract a compressed representation of context_text from layer L.

    Returns (context_vector [d_model] float32 CPU, n_context_tokens).
    """
    token_ids = wrapper.encode(context_text)
    n_tokens = len(token_ids)

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(wrapper.model.device)

    with torch.no_grad():
        outputs = wrapper.model(input_ids, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
    # index 0 = embedding output, index L+1 = output of layer L
    h = outputs.hidden_states[layer + 1][0]  # [seq_len, d_model]

    if pool == "mean":
        v = h.mean(dim=0)
    elif pool == "last":
        v = h[-1]
    elif pool == "cls":
        v = h[0]
    else:
        raise ValueError(f"Unknown pool: {pool!r}")

    return v.float().cpu(), n_tokens


# ---------------------------------------------------------------------------
# Context-injected generation
# ---------------------------------------------------------------------------

def generate_with_context_injection(
    wrapper: HFModelWrapper,
    question_text: str,
    context_vector: torch.Tensor,
    layer: int,
    scale: float = 1.0,
    max_new_tokens: int = 100,
    normalize: bool = True,
    stop_strings: tuple[str, ...] = QA_STOP_STRINGS,
) -> tuple[str, int]:
    """
    Generate response to question_text with context_vector injected at layer L.
    """
    if normalize:
        norm = context_vector.norm().item()
        if norm < _EPSILON:
            raise ValueError("context_vector has near-zero norm")
        ctx_v = context_vector / norm
    else:
        ctx_v = context_vector

    ctx_v = ctx_v.to(wrapper.model.device, dtype=torch.bfloat16)

    def hook_fn(module, input, output):
        # output is a tuple: (hidden_states, ...) for all decoder layer types
        hidden = output[0]
        hidden = hidden + scale * ctx_v[None, None, :]
        return (hidden,) + output[1:]

    handle = wrapper.layers[layer].register_forward_hook(hook_fn)

    token_ids = wrapper.encode(question_text)
    n_question_tokens = len(token_ids)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(wrapper.model.device)

    try:
        with torch.no_grad():
            out = wrapper.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=wrapper.eos_token_id,
            )
    finally:
        handle.remove()

    new_ids = out[0][len(token_ids):].tolist()
    text = wrapper.decode(new_ids)
    if stop_strings:
        text = _truncate_at_stop(text, stop_strings)
    return text, n_question_tokens


# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------

def generate_baseline_qa(
    wrapper: HFModelWrapper,
    full_prompt: str,
    max_new_tokens: int = 100,
    stop_strings: tuple[str, ...] = QA_STOP_STRINGS,
) -> tuple[str, int]:
    """Standard generation with full context in the prompt."""
    token_ids = wrapper.encode(full_prompt)
    n_input_tokens = len(token_ids)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(wrapper.model.device)

    with torch.no_grad():
        out = wrapper.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=wrapper.eos_token_id,
        )

    new_ids = out[0][len(token_ids):].tolist()
    text = wrapper.decode(new_ids)
    if stop_strings:
        text = _truncate_at_stop(text, stop_strings)
    return text, n_input_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_at_stop(text: str, stop_strings: tuple[str, ...]) -> str:
    best = len(text)
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1 and idx < best:
            best = idx
    return text[:best]
