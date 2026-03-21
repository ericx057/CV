from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_rollback_modal.backend_torch import select_runtime_device


@dataclass
class EHPCBundle:
    model_key: str
    model_name: str
    tokenizer: object
    model: object
    device: str
    dtype: torch.dtype
    n_layers: int
    n_heads: int


class AttentionCaptureComplete(RuntimeError):
    """Raised internally to stop a forward pass once required attention is captured."""


def load_bundle(model_key: str, model_name: str, eager_attention: bool = True) -> EHPCBundle:
    runtime = select_runtime_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "trust_remote_code": True,
        "dtype": runtime.dtype,
        "low_cpu_mem_usage": True,
    }
    if eager_attention:
        kwargs["attn_implementation"] = "eager"
    if runtime.device == "cuda" and torch.cuda.device_count() > 1:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not getattr(model, "hf_device_map", None):
        model.to(runtime.device)
    model.eval()
    model_inner = _resolve_model_inner(model)
    input_device = _resolve_input_device(model, model_inner)

    return EHPCBundle(
        model_key=model_key,
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=str(input_device),
        dtype=runtime.dtype,
        n_layers=int(model.config.num_hidden_layers),
        n_heads=int(model.config.num_attention_heads),
    )


def clear_bundle(bundle: EHPCBundle) -> None:
    del bundle.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_model_inner(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def _resolve_embed_tokens(model_inner):
    if hasattr(model_inner, "embed_tokens"):
        return model_inner.embed_tokens
    if hasattr(model_inner, "wte"):
        return model_inner.wte
    raise AttributeError("Unsupported architecture: could not find embedding layer")


def _resolve_input_device(model, model_inner) -> torch.device:
    try:
        embed = _resolve_embed_tokens(model_inner)
        weight = getattr(embed, "weight", None)
        if weight is not None:
            return weight.device
    except Exception:
        pass
    return next(model.parameters()).device


def get_decoder_layers(model) -> list[object]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError(
        f"Unsupported model architecture for EHPC attention capture: {type(model).__name__}"
    )


def run_attention_capture(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_callbacks: dict[int, Callable[[torch.Tensor], None]],
    stop_after_layer: int | None = None,
) -> None:
    layers = get_decoder_layers(model)
    originals: dict[int, Callable] = {}

    def make_wrapped_forward(layer_idx: int, original_forward: Callable):
        signature = inspect.signature(original_forward)

        def wrapped_forward(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            bound.arguments["output_attentions"] = True
            outputs = original_forward(*bound.args, **bound.kwargs)
            if not isinstance(outputs, tuple) or len(outputs) < 2 or outputs[1] is None:
                raise RuntimeError(
                    f"Layer {layer_idx} did not return attention weights during EHPC capture"
                )

            layer_callbacks[layer_idx](outputs[1])

            if stop_after_layer == layer_idx:
                raise AttentionCaptureComplete()
            return outputs

        return wrapped_forward

    try:
        for layer_idx in layer_callbacks:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise IndexError(
                    f"EHPC layer index {layer_idx} is out of range for {len(layers)} layers"
                )
            attention_module = getattr(layers[layer_idx], "self_attn", None)
            if attention_module is None:
                raise RuntimeError(
                    f"Decoder layer {layer_idx} has no self_attn module for EHPC capture"
                )
            originals[layer_idx] = attention_module.forward
            attention_module.forward = make_wrapped_forward(layer_idx, originals[layer_idx])

        with torch.no_grad():
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )
            except AttentionCaptureComplete:
                pass
    finally:
        for layer_idx, original_forward in originals.items():
            layers[layer_idx].self_attn.forward = original_forward
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def count_tokens(tokenizer, text: str) -> int:
    return len(encode_text(tokenizer, text))


def generate_from_prompt(bundle: EHPCBundle, prompt: str, max_new_tokens: int, stop_strings: tuple[str, ...] | None = None) -> str:
    inputs = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(bundle.device)
    with torch.no_grad():
        output = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=bundle.tokenizer.pad_token_id or bundle.tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if stop_strings:
        best = len(text)
        for stop in stop_strings:
            idx = text.find(stop)
            if idx != -1 and idx < best:
                best = idx
        text = text[:best]
    return text.strip()


def generate_from_ids(
    bundle: EHPCBundle,
    input_ids: list[int],
    max_new_tokens: int,
    stop_strings: tuple[str, ...] | None = None,
) -> str:
    tensor = torch.tensor([input_ids], device=bundle.device, dtype=torch.long)
    attention_mask = torch.ones_like(tensor)
    with torch.no_grad():
        output = bundle.model.generate(
            input_ids=tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=bundle.tokenizer.pad_token_id or bundle.tokenizer.eos_token_id,
        )
    new_tokens = output[0][tensor.shape[1]:]
    text = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if stop_strings:
        best = len(text)
        for stop in stop_strings:
            idx = text.find(stop)
            if idx != -1 and idx < best:
                best = idx
        text = text[:best]
    return text.strip()
