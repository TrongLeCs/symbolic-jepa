#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

echo "=== STARTING PIPELINE ==="

# Initialize Conda correctly
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Declare base directory based on script location to avoid hard-coding
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared artifacts directory for the entire pipeline.
# Only update the CASE_RESULT_DIR variable below for each case.
# Example: CASE_RESULT_DIR="case_encoder_ablation_01"
CASE_RESULT_DIR="case_07"
ARTIFACTS_DIR="$BASE_DIR/results/$CASE_RESULT_DIR"
mkdir -p "$ARTIFACTS_DIR"
export LOGIC_JEPA_ARTIFACTS_DIR="$ARTIFACTS_DIR"

echo "Artifacts dir: $LOGIC_JEPA_ARTIFACTS_DIR"

# ==========================================
# STEP 1: ENCODER
# ==========================================
echo "[1/4] Activating logic_jepa environment and running Encoder..."
# Changed to absolute path
conda activate /workspace/envs/logic_jepa
cd "$BASE_DIR/Encoder-Logic-JEPA"
python main.py

# ==========================================
# STEP 2: DECODER (MAIN)
# ==========================================
echo "[2/4] Running Decoder main.py..."
cd "$BASE_DIR/Decoder-Logic-JEPA"
python main.py

# ==========================================
# STEP 3: DECODER (INFERENCE)
# ==========================================
echo "[3/4] Running Inference process..."
python inference.py

# ==========================================
# STEP 4: EVALUATION (METRIC EVALUATION)
# ==========================================
echo "[4/4] Switching to env_metric environment and running evaluation..."
# NOTE: Replace the line below with your correct environment (e.g., conda activate repo_eval)
conda activate /root/miniconda3/envs/env_metric
cd "$BASE_DIR/metric_eval"
python main.py

echo "=== PIPELINE COMPLETED! ==="
echo "All artifacts are saved at: $LOGIC_JEPA_ARTIFACTS_DIR"