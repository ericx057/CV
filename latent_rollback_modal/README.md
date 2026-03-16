# latent_rollback_modal

Self-contained Modal-native port of the `latent_rollback` benchmark stack.

## What lives here

- Torch/Transformers runtime in `backend_torch.py`
- Modal orchestration in `modal_app.py`
- Modal-first CLI in `modal_cli.py`
- Copied benchmark entrypoints adapted for Linux/GPU execution

## Local smoke usage

```bash
python -m latent_rollback_modal.modal_cli repobench -- --models llama3-8b --n 1
```

## Modal usage

Run the same benchmark families through the functions exposed in `modal_app.py`:

- `repobench`
- `longbench`
- `matrix`
- `ablation`
- `refactor`
- `integration`

## Hardware

The Modal app is configured for `A100-80GB` so the large-model Person 2 path can carry `deepseek-14b` and `mistral-24b`.

## Person 2

RepoBench:

```bash
python -m latent_rollback_modal.modal_cli repobench -- \
  --models deepseek-14b mistral-24b \
  --n 50 \
  --split cross_file_first \
  --level 2k \
  --language python \
  --rank 8
```

Synthetic 17-task matrix:

```bash
python -m latent_rollback_modal.modal_cli integration -- \
  --models deepseek-14b mistral-24b \
  --tasks rename_3file add_param_chain interface_change move_function \
          who_calls type_provenance import_chain return_type_q param_type_q constant_q \
          transitive_return inherited_method field_access short_return short_constant \
          long_refactor long_double_hop \
  --passes 5
```
