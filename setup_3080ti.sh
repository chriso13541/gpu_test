#!/bin/bash
# =============================================================
# Setup script for 3080 Ti node
# Run this on your gaming PC
# =============================================================

set -e  # exit on any error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# -------------------------------------------------------------
# 1. Check prerequisites
# -------------------------------------------------------------
log "Checking prerequisites..."

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found. Are NVIDIA drivers installed?"
command -v nvcc       >/dev/null || fail "nvcc not found. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit"
command -v cmake      >/dev/null || fail "cmake not found: sudo apt install cmake"
command -v git        >/dev/null || fail "git not found: sudo apt install git"
command -v python3    >/dev/null || fail "python3 not found"
command -v pip3       >/dev/null || fail "pip3 not found: sudo apt install python3-pip"

CUDA_VER=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | awk '{print $1}')
VRAM_GB=$(echo "scale=1; $VRAM_MB/1024" | bc)

log "GPU:       $GPU_NAME"
log "VRAM:      ${VRAM_GB}GB"
log "CUDA:      $CUDA_VER"

# -------------------------------------------------------------
# 2. Build llama.cpp with CUDA
# -------------------------------------------------------------
INSTALL_DIR="$HOME/distributed-inference"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -d "llama.cpp" ]; then
    log "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
else
    log "llama.cpp already cloned, pulling latest..."
    cd llama.cpp && git pull && cd ..
fi

cd llama.cpp

log "Building llama.cpp with CUDA support..."
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=ON 2>&1 | tail -5

cmake --build build --config Release -j$(nproc) 2>&1 | tail -10

log "Build complete. Verifying..."
./build/bin/llama-cli --version || fail "Build failed"

cd "$INSTALL_DIR"

# -------------------------------------------------------------
# 3. Download model
# -------------------------------------------------------------
# Using Mistral 7B Q4_K_M — good quality/size tradeoff
# Fits in 4GB VRAM with room for context
MODEL_DIR="$INSTALL_DIR/models"
mkdir -p "$MODEL_DIR"

MODEL_FILE="$MODEL_DIR/mistral-7b-instruct-q4_k_m.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    log "Downloading Mistral 7B Q4_K_M (~4.4GB)..."
    # Using huggingface-cli if available, else wget
    if command -v huggingface-cli >/dev/null; then
        huggingface-cli download \
            bartowski/Mistral-7B-Instruct-v0.3-GGUF \
            Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
            --local-dir "$MODEL_DIR"
        mv "$MODEL_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" "$MODEL_FILE"
    else
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
else
    log "Model already downloaded, skipping."
fi

log "Model at: $MODEL_FILE"
MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
log "Model size: $MODEL_SIZE"

# -------------------------------------------------------------
# 4. Install Python deps for baseline runner
# -------------------------------------------------------------
log "Installing Python dependencies..."
pip3 install llama-cpp-python numpy psutil gputil --quiet

# -------------------------------------------------------------
# 5. Write node config
# -------------------------------------------------------------
cat > "$INSTALL_DIR/node_config.json" <<EOF
{
    "node_id": "3080ti-gaming-pc",
    "llama_cpp_bin": "$INSTALL_DIR/llama.cpp/build/bin/llama-cli",
    "model_path": "$MODEL_FILE",
    "total_vram_mb": $VRAM_MB,
    "gpu_name": "$GPU_NAME",
    "cuda_version": "$CUDA_VER",
    "experiments": {
        "A": {"label": "8GB on 3080Ti",  "gpu_layers": 28},
        "B": {"label": "4GB on 3080Ti",  "gpu_layers": 14},
        "C": {"label": "CPU only",        "gpu_layers": 0}
    }
}
EOF

log "Config written to $INSTALL_DIR/node_config.json"
log ""
log "Setup complete. Next step:"
log "  cd $INSTALL_DIR && python3 scripts/baseline_runner.py"
