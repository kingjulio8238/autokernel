#!/usr/bin/env bash
# setup.sh — Full setup for a fresh GPU pod from scratch.
# Usage: PAT=ghp_xxx bash setup.sh
#   or:  bash setup.sh  (if repo is already cloned)
set -euo pipefail

REPO_DIR="/workspace/autokernel"

# === 1. Clone repo (skip if already cloned) ===
if [ ! -d "$REPO_DIR/.git" ]; then
    if [ -z "${PAT:-}" ]; then
        echo "ERROR: Set PAT env var for git clone. Usage: PAT=ghp_xxx bash setup.sh"
        exit 1
    fi
    echo "=== 1. Cloning repo ==="
    git clone "https://${PAT}@github.com/kingjulio8238/autokernel.git" "$REPO_DIR"
else
    echo "=== 1. Repo already cloned ==="
fi

cd "$REPO_DIR"

# === 2. Checkout working branch ===
echo "=== 2. Checkout autokernel/mar12 ==="
git checkout autokernel/mar12 2>/dev/null || echo "  already on $(git branch --show-current)"

# === 3. System deps ===
echo "=== 3. System deps ==="
apt-get update -qq && apt-get install -y -qq python3.10-dev > /dev/null
echo "  python3.10-dev installed"

# === 4. Install uv (if missing) ===
echo "=== 4. Install uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

# === 5. Python deps ===
echo "=== 5. Python deps ==="
uv sync
echo "  uv sync done"

# === 6. Verify ===
echo "=== 6. Verify ==="
uv run python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
uv run python -c "from kernelbench.eval import eval_kernel_against_ref; print('  KernelBench: OK')"

# === 7. Smoke test ===
echo "=== 7. Smoke test (prepare.py) ==="
uv run python prepare.py

echo ""
echo "=== Setup complete. Ready to iterate. ==="
