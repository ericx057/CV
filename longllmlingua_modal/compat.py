from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompressorCompat:
    accepts_var_kwargs: bool
    accepted_params: set[str]
    ratio_param: str | None
    target_param: str | None


@dataclass(frozen=True)
class CompressionPayload:
    compressed_prompt: str
    origin_tokens: int | None
    compressed_tokens: int | None
    ratio: str | float | None
    raw: dict[str, Any]


def patch_prompt_compressor_cache_compat(compressor) -> None:
    original_get_ppl = getattr(compressor, "get_ppl", None)
    if original_get_ppl is None:
        return
    if getattr(compressor, "_cv_cache_compat_patched", False):
        return

    patch_prompt_compressor_model_cache_compat(compressor)

    def wrapped_get_ppl(*args, **kwargs):
        if "past_key_values" in kwargs:
            kwargs["past_key_values"] = normalize_past_key_values(kwargs["past_key_values"])
        result = original_get_ppl(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            score, past_key_values = result
            return score, normalize_past_key_values(past_key_values)
        return result

    compressor.get_ppl = wrapped_get_ppl
    setattr(compressor, "_cv_cache_compat_patched", True)


def patch_prompt_compressor_model_cache_compat(compressor) -> None:
    model = getattr(compressor, "model", None)
    original_forward = getattr(model, "forward", None)
    if model is None or original_forward is None:
        return
    if getattr(model, "_cv_forward_cache_compat_patched", False):
        return

    def wrapped_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        return normalize_model_output_cache(outputs)

    model.forward = wrapped_forward
    setattr(model, "_cv_forward_cache_compat_patched", True)


def infer_compressor_compat(compress_prompt_fn) -> CompressorCompat:
    sig = inspect.signature(compress_prompt_fn)
    params = sig.parameters
    accepted = set(params)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )

    if "rate" in accepted or accepts_var_kwargs:
        ratio_param = "rate"
    elif "ratio" in accepted:
        ratio_param = "ratio"
    else:
        ratio_param = None

    if "target_token" in accepted or accepts_var_kwargs:
        target_param = "target_token"
    elif "target_tokens" in accepted:
        target_param = "target_tokens"
    else:
        target_param = None

    return CompressorCompat(
        accepts_var_kwargs=accepts_var_kwargs,
        accepted_params=accepted,
        ratio_param=ratio_param,
        target_param=target_param,
    )


def normalize_compress_kwargs(
    compat: CompressorCompat,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(kwargs)

    if "rate" in normalized and compat.ratio_param and compat.ratio_param != "rate":
        normalized[compat.ratio_param] = normalized.pop("rate")
    elif "ratio" in normalized and compat.ratio_param and compat.ratio_param != "ratio":
        normalized[compat.ratio_param] = normalized.pop("ratio")

    if (
        "target_token" in normalized
        and compat.target_param
        and compat.target_param != "target_token"
    ):
        normalized[compat.target_param] = normalized.pop("target_token")
    elif (
        "target_tokens" in normalized
        and compat.target_param
        and compat.target_param != "target_tokens"
    ):
        normalized[compat.target_param] = normalized.pop("target_tokens")

    if compat.accepts_var_kwargs:
        return normalized

    return {
        key: value
        for key, value in normalized.items()
        if key in compat.accepted_params
    }


def extract_compression_payload(result: Any) -> CompressionPayload:
    if not isinstance(result, dict):
        raise TypeError(
            f"Unexpected LongLLMLingua response type: {type(result).__name__}"
        )

    prompt = None
    for key in ("compressed_prompt", "compressed_text", "prompt", "text"):
        value = result.get(key)
        if isinstance(value, str) and value:
            prompt = value
            break

    if prompt is None:
        raise KeyError(
            "LongLLMLingua result did not contain a supported compressed prompt key"
        )

    origin_tokens = _first_numeric(result, "origin_tokens", "origin_token_count")
    compressed_tokens = _first_numeric(
        result, "compressed_tokens", "compressed_token_count"
    )
    ratio = result.get("ratio")

    return CompressionPayload(
        compressed_prompt=prompt,
        origin_tokens=origin_tokens,
        compressed_tokens=compressed_tokens,
        ratio=ratio,
        raw=result,
    )


def normalize_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    if not isinstance(past_key_values, (list, tuple)):
        return past_key_values

    normalized_layers: list[tuple[Any, Any]] = []
    for layer_cache in past_key_values:
        if not isinstance(layer_cache, (list, tuple)):
            return past_key_values
        if len(layer_cache) < 2:
            return past_key_values
        key, value = layer_cache[0], layer_cache[1]
        normalized_layers.append((key, value))
    return tuple(normalized_layers)


def normalize_model_output_cache(outputs: Any) -> Any:
    if outputs is None:
        return outputs

    if hasattr(outputs, "past_key_values"):
        normalized = normalize_past_key_values(getattr(outputs, "past_key_values"))
        try:
            setattr(outputs, "past_key_values", normalized)
        except Exception:
            pass
        try:
            outputs["past_key_values"] = normalized
        except Exception:
            pass
        return outputs

    if isinstance(outputs, tuple) and len(outputs) >= 2:
        normalized = normalize_past_key_values(outputs[1])
        return (outputs[0], normalized, *outputs[2:])

    return outputs


def _first_numeric(result: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None
