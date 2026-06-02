#!/bin/bash
# =============================================================
# Setup script for ThinkPad P53 — Quadro T1000 4GB node
# Same as 3080Ti setup but with T1000-specific config
# =============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

log "Setting up T1000 node (ThinkPad P53)..."

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
command -v nvcc       >/dev/null || fail "nvcc not found. Install: sudo apt install nvidia-cuda-toolkit"
command -v cmake      >/dev/null || fail "cmake not found: sudo apt install cmake"
command -v git        >/dev/null || fail "git not found"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | awk '{print $1}')
log "GPU:  $GPU_NAME"
log "VRAM: ${VRAM_MB}MB"

INSTALL_DIR="$HOME/distributed-inference"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -d "llama.cpp" ]; then
    log "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
else
    cd llama.cpp && git pull && cd ..
fi

cd llama.cpp
log "Building with CUDA..."
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build build --config Release -j$(nproc) 2>&1 | tail -5
cd "$INSTALL_DIR"

# T1000 has 4GB — use same model as 3080Ti for apples-to-apples comparison
MODEL_DIR="$INSTALL_DIR/models"
mkdir -p "$MODEL_DIR"
MODEL_FILE="$MODEL_DIR/mistral-7b-instruct-q4_k_m.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    log "Downloading Mistral 7B Q4_K_M (~4.4GB)..."
    pip3 install huggingface_hub --quiet
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/Mistral-7B-Instruct-v0.3-GGUF',
    filename='Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
    local_dir='$MODEL_DIR'
)
"
    mv "$MODEL_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" "$MODEL_FILE"
fi

# T1000 config:
# - Experiment C matches 3080Ti Experiment B (same 14 GPU layers, different bandwidth)
# - Experiment D is T1000 fully loaded (all layers that fit in 4GB)
# - Experiment E is CPU-only fallback
cat > "$INSTALL_DIR/node_config.json" <<EOF
{
    "node_id": "t1000-thinkpad-p53",
    "llama_cpp_bin": "$INSTALL_DIR/llama.cpp/build/bin/llama-cli",
    "model_path": "$MODEL_FILE",
    "total_vram_mb": $VRAM_MB,
    "gpu_name": "$GPU_NAME",
    "experiments": {
        "C": {"label": "4GB T1000 (14 layers)",  "gpu_layers": 14},
        "D": {"label": "4GB T1000 (max layers)",  "gpu_layers": 32},
        "E": {"label": "CPU only (T1000 baseline)", "gpu_layers": 0}
    }
}
EOF

# Copy baseline runner
cp "$(dirname "$0")/baseline_runner.py" "$INSTALL_DIR/scripts/" 2>/dev/null || true

log ""
log "Setup complete."
log "Copy baseline_runner.py to this machine then run:"
log "  python3 $INSTALL_DIR/scripts/baseline_runner.py"
