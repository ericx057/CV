#!/usr/bin/env bash
# Setup script for M4 Apple Silicon (macOS 14+ / Sequoia)
# Creates a venv, installs PyTorch with MPS support, then TransformerLens.
#
# Run once:
#   chmod +x setup_m4.sh && ./setup_m4.sh

set -euo pipefail

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

echo "==> Creating virtual environment at $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

# PyTorch 2.3+ has solid MPS support for bfloat16 and most ops.
# The default pip install of torch on macOS automatically uses MPS.
echo "==> Installing PyTorch (MPS-enabled via default macOS wheel)"
pip install torch torchvision torchaudio

echo "==> Installing project dependencies"
pip install -r requirements.txt

echo ""
echo "==> Verifying MPS availability"
python3 - <<'PYEOF'
import torch
mps_avail  = torch.backends.mps.is_available()
mps_built  = torch.backends.mps.is_built()
print(f"  torch version : {torch.__version__}")
print(f"  MPS available : {mps_avail}")
print(f"  MPS built     : {mps_built}")
if mps_avail and mps_built:
    t = torch.zeros(3, device="mps")
    print(f"  MPS smoke test: PASS  (tensor on mps: {t.device})")
else:
    print("  MPS not available — experiment will fall back to CPU")
PYEOF

echo ""
echo "==> Setup complete."
echo ""
echo "Activate env with:  source $VENV_DIR/bin/activate"
echo ""
echo "Quick smoke test (GPT-2 XL, no auth required):"
echo "  python run_experiment.py --model gpt2xl"
echo ""
echo "Primary experiment (Mistral 7B, ~14 GB RAM):"
echo "  python run_experiment.py --model mistral"
echo ""
echo "Llama 3 8B (requires HuggingFace token):"
echo "  export HF_TOKEN=hf_<your_token>"
echo "  python run_experiment.py --model llama3"
echo ""
echo "With ablation sweep:"
echo "  python run_experiment.py --model mistral --ablation"
