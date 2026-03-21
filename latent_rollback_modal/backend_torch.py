from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

_EPSILON = 1e-8


@dataclass(frozen=True)
class RuntimeDevice:
    device: str
    dtype: torch.dtype


def select_runtime_device() -> RuntimeDevice:
    if torch.cuda.is_available():
        return RuntimeDevice(device="cuda", dtype=torch.bfloat16)
    return RuntimeDevice(device="cpu", dtype=torch.float32)


class TorchModelWrapper:
    """Thin wrapper around a Hugging Face causal LM."""

    def __init__(self, model, tokenizer, model_name: str):
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._model_key = model_name
        self._inner = _resolve_model_inner(model)
        self._layers = _resolve_layers(self._inner)
        self.n_layers = len(self._layers)
        self.d_model = int(_resolve_embed_tokens(self._inner).weight.shape[1])
        self.device = _resolve_input_device(model, self._inner)

    def encode(self, prompt: str) -> list[int]:
        return self._tokenizer.encode(prompt, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self._tokenizer, "eos_token_id", None)

    @property
    def layers(self):
        return self._layers

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        input_ids = _tensorize(token_ids, self.device)
        return _resolve_embed_tokens(self._inner)(input_ids)


MLXModelWrapper = TorchModelWrapper


def load_model(model_name: str) -> TorchModelWrapper:
    runtime = select_runtime_device()
    console.print(f"[bold cyan]Loading Torch model:[/bold cyan] {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except Exception as exc:
        console.print(
            f"[yellow]Fast tokenizer load failed for {model_name}; "
            f"retrying with use_fast=False ({type(exc).__name__}: {exc})[/yellow]"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": runtime.dtype,
        "low_cpu_mem_usage": True,
    }
    if runtime.device == "cuda" and torch.cuda.device_count() > 1:
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if not getattr(model, "hf_device_map", None):
        model.to(runtime.device)
    model.eval()
    wrapper = TorchModelWrapper(model, tokenizer, model_name)
    console.print(
        f"[green]Model loaded.[/green] "
        f"Layers: {wrapper.n_layers}  d_model: {wrapper.d_model}  device: {runtime.device}"
    )
    return wrapper


def clear_backend_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_model_inner(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def _resolve_layers(model_inner):
    if hasattr(model_inner, "layers"):
        return model_inner.layers
    if hasattr(model_inner, "h"):
        return model_inner.h
    raise AttributeError("Unsupported architecture: could not find decoder layers")


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


def _tensorize(token_ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], device=device, dtype=torch.long)


def _split_layer_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _merge_layer_output(hidden_states, tail):
    if tail is None:
        return hidden_states
    return (hidden_states, *tail)


def _run_layers(
    wrapper: TorchModelWrapper,
    token_ids: list[int],
    hook_layer: int | None = None,
    hook_fn=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_ids = _tensorize(token_ids, wrapper.device)
    attention_mask = torch.ones_like(input_ids)
    captured: dict[str, torch.Tensor] = {}
    handle = None

    if hook_layer is not None:
        layer_module = wrapper.layers[hook_layer]

        def _forward_hook(_module, _args, output):
            hidden_states, tail = _split_layer_output(output)
            captured["h"] = hidden_states.detach()
            if hook_fn is None:
                return output
            modified = hook_fn(hidden_states)
            return _merge_layer_output(modified, tail)

        handle = layer_module.register_forward_hook(_forward_hook)

    try:
        with torch.no_grad():
            outputs = wrapper._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        if handle is not None:
            handle.remove()

    return outputs.logits, captured.get("h")


def extract_layer_hidden_states(
    wrapper: TorchModelWrapper,
    token_ids: list[int],
    layer: int,
) -> torch.Tensor:
    input_ids = _tensorize(token_ids, wrapper.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = wrapper._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return outputs.hidden_states[layer + 1].detach()


def extract_residual_vector(
    wrapper: TorchModelWrapper,
    prompt: str,
    layer: int,
    token_position: int = -1,
) -> tuple[torch.Tensor, str]:
    token_ids = wrapper.encode(prompt)
    actual_pos = token_position % len(token_ids)
    logits, h_at_hook = _run_layers(wrapper, token_ids, hook_layer=layer)
    if h_at_hook is None:
        raise ValueError(f"Failed to capture hidden state at layer {layer}")
    vector = h_at_hook[0, actual_pos, :].detach().float().cpu()
    next_token_id = int(torch.argmax(logits[0, -1, :]).item())
    return vector, wrapper.decode([next_token_id])


def extract_all_layer_vectors(
    wrapper: TorchModelWrapper,
    prompt: str,
    token_position: int = -1,
) -> list[torch.Tensor]:
    token_ids = wrapper.encode(prompt)
    actual_pos = token_position % len(token_ids)
    input_ids = _tensorize(token_ids, wrapper.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = wrapper._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return [
        hidden[0, actual_pos, :].detach().float().cpu()
        for hidden in outputs.hidden_states[1:]
    ]


def generate_baseline(
    wrapper: TorchModelWrapper,
    token_ids: list[int],
    max_new_tokens: int = 80,
) -> list[int]:
    tokens = list(token_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits, _ = _run_layers(wrapper, tokens)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break
    return generated


def generate_with_matrix_hook(
    wrapper: TorchModelWrapper,
    token_ids: list[int],
    layer_matrices: dict[int, tuple[torch.Tensor, torch.Tensor]],
    mode: str = "rollback",
    scale: float = 1.0,
    max_new_tokens: int = 80,
    broadcast: bool = True,
) -> list[int]:
    prompt_len = len(token_ids)
    sign = -1.0 if mode == "rollback" else 1.0
    cached = {
        layer: (
            A.to(device=wrapper.device, dtype=next(wrapper._model.parameters()).dtype),
            B.to(device=wrapper.device, dtype=next(wrapper._model.parameters()).dtype),
        )
        for layer, (A, B) in layer_matrices.items()
    }

    def run_forward(tokens: list[int]) -> torch.Tensor:
        current_len = len(tokens)
        hook_handles = []
        try:
            for layer, (A, B) in cached.items():
                module = wrapper.layers[layer]

                def _make_hook(A_local, B_local):
                    def _hook(_module, _args, output):
                        hidden_states, tail = _split_layer_output(output)
                        active_len = min(prompt_len if broadcast else current_len, hidden_states.shape[1])
                        if active_len <= 0:
                            return output
                        if broadcast:
                            target = hidden_states[:, :active_len, :]
                            proj = torch.matmul(target, A_local.transpose(0, 1))
                            correction = torch.matmul(proj, B_local.transpose(0, 1))
                            hidden_states = hidden_states.clone()
                            hidden_states[:, :active_len, :] = target + (sign * scale) * correction
                            return _merge_layer_output(hidden_states, tail)
                        target_pos = min(prompt_len - 1, hidden_states.shape[1] - 1)
                        target = hidden_states[:, target_pos:target_pos + 1, :]
                        proj = torch.matmul(target, A_local.transpose(0, 1))
                        correction = torch.matmul(proj, B_local.transpose(0, 1))
                        hidden_states = hidden_states.clone()
                        hidden_states[:, target_pos:target_pos + 1, :] = target + (sign * scale) * correction
                        return _merge_layer_output(hidden_states, tail)
                    return _hook

                hook_handles.append(module.register_forward_hook(_make_hook(A, B)))

            logits, _ = _run_layers(wrapper, tokens)
            return logits
        finally:
            for handle in hook_handles:
                handle.remove()

    tokens = list(token_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = run_forward(tokens)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break
    return generated


def _ids_to_str(wrapper: TorchModelWrapper, ids: list[int]) -> str:
    if not ids:
        console.print("[yellow]Warning: generation produced zero tokens.[/yellow]")
        return ""
    return wrapper.decode(ids)
