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
