from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .config import head_config_path, packaged_head_config_path, resolve_model_key
from .model_utils import run_attention_capture


@dataclass
class CompressionResult:
    compressed_prompt: str
    compressed_context: str
    compressed_input_ids: list[int]
    original_input_tokens: int
    compressed_input_tokens: int
    original_context_tokens: int
    compressed_context_tokens: int
    observation_window_tokens: int
    retained_token_indices: list[int]
    attention_estimated_gb: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def load_head_config(model_key: str) -> dict:
    volume_path = head_config_path(model_key)
    package_path = packaged_head_config_path(model_key)
    path = volume_path if volume_path.exists() else package_path
    if not path.exists():
        resolved = resolve_model_key(model_key)
        raise FileNotFoundError(
            f"No evaluator head config found for {resolved}. "
            f"Run: python -m ehpc_modal.modal_cli pilot --model {resolved}"
        )
    with open(path) as handle:
        return json.load(handle)


def compute_token_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    evaluator_layer: int,
    evaluator_heads: list[int],
    observation_window: int,
    pool_kernel: int = 3,
    max_attention_gb: float | None = None,
) -> torch.Tensor:
    estimated_gb = estimate_attention_gb(model, input_ids.shape[1], layer_count=1)
    if max_attention_gb is not None and estimated_gb > max_attention_gb:
        raise RuntimeError(
            "EHPC attention capture budget exceeded: "
            f"estimated={estimated_gb:.2f}GiB budget={max_attention_gb:.2f}GiB "
            f"seq_len={input_ids.shape[1]}"
        )

    score_state: dict[str, torch.Tensor] = {}

    def capture_scores(layer_attn: torch.Tensor) -> None:
        attn_at_layer = layer_attn[0]
        seq_len = attn_at_layer.shape[-1]
        local_observation_window = max(1, min(observation_window, seq_len))
        start = max(0, seq_len - local_observation_window)
        total_scores = torch.zeros(seq_len, device=attn_at_layer.device, dtype=torch.float32)

        for head_idx in evaluator_heads:
            obs_attn = attn_at_layer[head_idx, start:seq_len, :].float()
            head_scores = obs_attn.sum(dim=0) / local_observation_window
            pooled = F.avg_pool1d(
                head_scores.unsqueeze(0).unsqueeze(0),
                kernel_size=pool_kernel,
                stride=1,
                padding=pool_kernel // 2,
            ).squeeze(0).squeeze(0)
            total_scores += pooled

        score_state["scores"] = total_scores.detach().cpu()

    run_attention_capture(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layer_callbacks={evaluator_layer: capture_scores},
        stop_after_layer=evaluator_layer,
    )

    if "scores" not in score_state:
        raise RuntimeError(
            f"EHPC failed to capture evaluator-layer attention at layer {evaluator_layer}"
        )
    return score_state["scores"]


def _build_prompt(prefix_text: str, context_text: str, suffix_text: str) -> str:
    return f"{prefix_text}{context_text}{suffix_text}"


def estimate_attention_gb(model, seq_len: int, layer_count: int | None = None) -> float:
    n_layers = (
        int(layer_count)
        if layer_count is not None
        else int(getattr(model.config, "num_hidden_layers"))
    )
    n_heads = int(getattr(model.config, "num_attention_heads"))
    dtype = next(model.parameters()).dtype
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = n_layers * n_heads * seq_len * seq_len * element_size
    return total_bytes / (1024 ** 3)


def _tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError(
            "EHPC requires a fast tokenizer with offset mapping support for exact token accounting"
        )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list) and len(offsets) == 1:
        offsets = offsets[0]
    return list(input_ids), [(int(start), int(end)) for start, end in offsets]


def _split_prompt_indices(
    offsets: list[tuple[int, int]],
    context_start_char: int,
    context_end_char: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    prefix_indices: list[int] = []
    protected_context_indices: list[int] = []
    selectable_context_indices: list[int] = []
    suffix_indices: list[int] = []

    for idx, (start, end) in enumerate(offsets):
        overlaps_context = end > context_start_char and start < context_end_char
        fully_inside_context = start >= context_start_char and end <= context_end_char

        if overlaps_context:
            if fully_inside_context:
                selectable_context_indices.append(idx)
            else:
                # Boundary-crossing tokens are always retained so prompt scaffolding
                # does not disappear when context tokens are removed.
                protected_context_indices.append(idx)
        elif end <= context_start_char:
            prefix_indices.append(idx)
        elif start >= context_end_char:
            suffix_indices.append(idx)

    return prefix_indices, protected_context_indices, selectable_context_indices, suffix_indices


def compress_prompt(
    *,
    model,
    tokenizer,
    prefix_text: str,
    context_text: str,
    suffix_text: str,
    head_config: dict,
    target_tokens: int | None = None,
    compression_ratio: float | None = None,
    observation_window_tokens: int | None = None,
    pool_kernel: int = 3,
    max_attention_gb: float | None = None,
    device: str = "cuda",
) -> CompressionResult:
    if target_tokens is None and compression_ratio is None:
        raise ValueError("Must specify target_tokens or compression_ratio")

    full_prompt = _build_prompt(prefix_text, context_text, suffix_text)
    full_ids, offsets = _tokenize_with_offsets(tokenizer, full_prompt)
    context_start_char = len(prefix_text)
    context_end_char = context_start_char + len(context_text)
    (
        prefix_indices,
        protected_context_indices,
        selectable_context_indices,
        suffix_indices,
    ) = _split_prompt_indices(offsets, context_start_char, context_end_char)
    seq_len = len(full_ids)
    context_token_indices = protected_context_indices + selectable_context_indices
    original_context_tokens = len(context_token_indices)

    if original_context_tokens == 0:
        raise ValueError("No context tokens identified for EHPC compression")

    if target_tokens is not None:
        n_keep = max(1, min(target_tokens, original_context_tokens))
    else:
        n_keep = max(1, min(original_context_tokens, int(original_context_tokens * compression_ratio)))

    if observation_window_tokens is None:
        suffix_tokens = len(suffix_indices)
        observation_window_tokens = max(1, suffix_tokens)

    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    scores = compute_token_scores(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        evaluator_layer=int(head_config["evaluator_layer"]),
        evaluator_heads=list(head_config["evaluator_heads"]),
        observation_window=observation_window_tokens,
        pool_kernel=pool_kernel,
        max_attention_gb=max_attention_gb,
    )

    if n_keep >= original_context_tokens:
        keep_context_indices = sorted(context_token_indices)
    else:
        selectable_keep = max(0, n_keep - len(protected_context_indices))
        if selectable_keep >= len(selectable_context_indices):
            selected_context_indices = selectable_context_indices
        elif selectable_keep == 0:
            selected_context_indices = []
        else:
            context_scores = scores[selectable_context_indices]
            topk = torch.topk(context_scores, k=selectable_keep, largest=True, sorted=False)
            selected_context_indices = [
                selectable_context_indices[idx]
                for idx in sorted(topk.indices.tolist())
            ]
        keep_context_indices = sorted(protected_context_indices + selected_context_indices)

    compressed_input_ids = [full_ids[idx] for idx in prefix_indices]
    compressed_input_ids.extend(full_ids[idx] for idx in keep_context_indices)
    compressed_input_ids.extend(full_ids[idx] for idx in suffix_indices)
    kept_context_ids = [full_ids[idx] for idx in keep_context_indices]
    compressed_context = tokenizer.decode(kept_context_ids, skip_special_tokens=True)
    compressed_prompt = tokenizer.decode(compressed_input_ids, skip_special_tokens=True)
    compressed_input_tokens = len(compressed_input_ids)

    return CompressionResult(
        compressed_prompt=compressed_prompt,
        compressed_context=compressed_context,
        compressed_input_ids=compressed_input_ids,
        original_input_tokens=seq_len,
        compressed_input_tokens=compressed_input_tokens,
        original_context_tokens=original_context_tokens,
        compressed_context_tokens=len(kept_context_ids),
        observation_window_tokens=observation_window_tokens,
        retained_token_indices=keep_context_indices,
        attention_estimated_gb=estimate_attention_gb(model, seq_len, layer_count=1),
    )


def compress_qa_prompt(
    *,
    model,
    tokenizer,
    context_text: str,
    question_text: str,
    head_config: dict,
    target_tokens: int | None = None,
    compression_ratio: float | None = None,
    pool_kernel: int = 3,
    max_attention_gb: float | None = None,
    device: str = "cuda",
) -> CompressionResult:
    prefix = "Context:\n"
    suffix = f"\n\nQuestion: {question_text}\n\nAnswer:"
    question_tokens = len(tokenizer.encode(question_text, add_special_tokens=False))
    observation_window = question_tokens + 8
    return compress_prompt(
        model=model,
        tokenizer=tokenizer,
        prefix_text=prefix,
        context_text=context_text,
        suffix_text=suffix,
        head_config=head_config,
        target_tokens=target_tokens,
        compression_ratio=compression_ratio,
        observation_window_tokens=observation_window,
        pool_kernel=pool_kernel,
        max_attention_gb=max_attention_gb,
        device=device,
    )


def compress_code_prompt(
    *,
    model,
    tokenizer,
    cross_file_context: str,
    local_context: str,
    head_config: dict,
    target_tokens: int | None = None,
    compression_ratio: float | None = None,
    pool_kernel: int = 3,
    max_attention_gb: float | None = None,
    device: str = "cuda",
) -> CompressionResult:
    prefix = "# Cross-file context:\n"
    suffix = f"\n\n# Current file:\n{local_context}\n\n# Complete the next line:\n"
    code_query = local_context[-200:] if len(local_context) > 200 else local_context
    observation_window = max(8, len(tokenizer.encode(code_query, add_special_tokens=False)))
    return compress_prompt(
        model=model,
        tokenizer=tokenizer,
        prefix_text=prefix,
        context_text=cross_file_context,
        suffix_text=suffix,
        head_config=head_config,
        target_tokens=target_tokens,
        compression_ratio=compression_ratio,
        observation_window_tokens=observation_window,
        pool_kernel=pool_kernel,
        max_attention_gb=max_attention_gb,
        device=device,
    )
