#!/bin/bash
# =============================================================================
# run_ssl_training.sh — Stage 1: SSL Self-Supervised Pre-training
#
# Trains image and fluorescence encoders using contrastive learning on
# UNLABELED holographic microscopy data.
#
# Usage:
#   bash run_ssl_training.sh                                  # default config
#   bash run_ssl_training.sh /path/to/custom_config.yaml      # one config file
#   bash run_ssl_training.sh /path/to/config_dir/             # BATCH: every *.yaml / *.yml in the dir
#
# Before running, edit the GPU CONFIGURATION block below and set the
# following fields in your config YAML:
#   data.dataset_path        — path to your unlabeled HDF5 file  (CHANGE ME)
#   logging.checkpoint_dir   — where to save SSL checkpoints     (CHANGE ME)
# =============================================================================

set -e

# --- Colour helpers ----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[1;34m'; CYAN='\033[1;36m'; MAGENTA='\033[1;35m'
BOLD='\033[1m'; NC='\033[0m'
ok()    { echo -e "${GREEN}✓${NC} $1"; }
info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Banner ------------------------------------------------------------------
cat <<'EOF'

 ██████╗ ██╗ ██████╗      █████╗ ██╗██████╗ ███╗   ███╗███████╗████████╗
 ██╔══██╗██║██╔═══██╗    ██╔══██╗██║██╔══██╗████╗ ████║██╔════╝╚══██╔══╝
 ██████╔╝██║██║   ██║    ███████║██║██████╔╝██╔████╔██║█████╗     ██║   
 ██╔══██╗██║██║   ██║    ██╔══██║██║██╔══██╗██║╚██╔╝██║██╔══╝     ██║   
 ██████╔╝██║╚██████╔╝    ██║  ██║██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   
 ╚═════╝ ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝

                   Stage 1 — SSL Self-Supervised Pre-training

EOF
echo -e "${BOLD}${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}   BioAirMet SSL Training - PTB, NPL                               ${NC}"
echo -e "${BOLD}${MAGENTA}════════════════════════════════════════════════════════════════════${NC}\n"

# =============================================================================
# GPU CONFIGURATION
# Choose ONE of the two modes below (comment out the other).
# The choice here must match your config YAML (distributed.*).
# =============================================================================

# --- Option A: Single GPU ----------------------------------------------------
# Uses one GPU. In your config YAML set:
#   distributed.enable: True
#   distributed.world_size: 1
#   distributed.gpu_ids: [0]        ← change to the GPU index you want
# GPU_MODE="single"
# GPU_IDS="0"                          # CHANGE ME — single GPU index

# --- Option B: Multi-GPU (DDP) -----------------------------------------------
# Uses multiple GPUs via PyTorch DDP. In your config YAML set:
#   distributed.enable: True
#   distributed.world_size: -1      ← -1 = all visible GPUs
#   distributed.gpu_ids: []         ← [] = all visible GPUs
# Uncomment the three lines below and comment out Option A:
GPU_MODE="multi"
GPU_IDS="0,1"                 # CHANGE ME — comma-separated GPU indices
#                                   # Must match distributed.gpu_ids in config

# =============================================================================
# END OF GPU CONFIGURATION
# =============================================================================

# Apply GPU visibility
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# --- Script config -----------------------------------------------------------
CONFIG_PATH="${1:-src/bioairmet/config/ssl/SSL_config_general.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Environment setup -------------------------------------------------------
if [ -n "$VIRTUAL_ENV" ]; then
    info "Environment already active: ${VIRTUAL_ENV}"
elif [ -n "${ENV_PATH:-}" ] && [ -f "${ENV_PATH}/bin/activate" ]; then
    source "${ENV_PATH}/bin/activate"
    ok "Activated: ${VIRTUAL_ENV}"
else
    for _c in "${SCRIPT_DIR}/.venv" "${SCRIPT_DIR}/venv" "${SCRIPT_DIR}/env"; do
        if [ -f "${_c}/bin/activate" ]; then
            source "${_c}/bin/activate"; ok "Activated: ${VIRTUAL_ENV}"; break
        fi
    done
    [ -z "$VIRTUAL_ENV" ] && warn "No venv found — using system Python. Set ENV_PATH if needed."
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# --- Pre-flight checks -------------------------------------------------------
# The argument may be EITHER a single config file OR a directory:
#   - a file      -> train just that config
#   - a directory -> BATCH: train every *.yaml / *.yml in it (sorted by name)
[ -e "$CONFIG_PATH" ] || error "Path not found: $CONFIG_PATH
  Pass a config file, or a directory containing one or more configs:
    bash run_ssl_training.sh /path/to/config.yaml
    bash run_ssl_training.sh /path/to/config_dir/
  Default: src/bioairmet/config/ssl/SSL_config_general.yaml"

CONFIG_FILES=()
BATCH_MODE=false
if [ -d "$CONFIG_PATH" ]; then
    while IFS= read -r -d '' _f; do
        CONFIG_FILES+=("$_f")
    done < <(find "$CONFIG_PATH" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 | sort -z)
    [ "${#CONFIG_FILES[@]}" -gt 0 ] || error "No config files (*.yaml / *.yml) found in directory: $CONFIG_PATH"
    BATCH_MODE=true
else
    [ -f "$CONFIG_PATH" ] || error "Not a config file or a directory: $CONFIG_PATH"
    CONFIG_FILES=("$CONFIG_PATH")
fi

# Determine visible GPU count
if command -v nvidia-smi &>/dev/null; then
    N_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
else
    N_GPUS="? (nvidia-smi not found)"
fi

echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}Configuration Summary${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
if [ "${BATCH_MODE}" = true ]; then
    echo -e "${BLUE}Configs (batch)    :${NC} ${#CONFIG_FILES[@]} file(s) from ${CONFIG_PATH}"
    for _f in "${CONFIG_FILES[@]}"; do echo -e "${BLUE}    • ${NC}${_f}"; done
else
    echo -e "${BLUE}Config file        :${NC} ${CONFIG_PATH}"
fi
echo -e "${BLUE}Mode               :${NC} ssl"
echo -e "${BLUE}GPU IDs          :${NC} ${GPU_IDS}"

if [ "$GPU_MODE" = "multi" ]; then
    echo -e "${BLUE}GPU mode           :${NC} ${GREEN}MULTI-GPU (DDP)${NC}"
    echo -e "${BLUE}CUDA_VISIBLE_DEVICES:${NC} ${GPU_IDS}  (${N_GPUS} GPU(s))"
    echo -e "${YELLOW}  ↳ Ensure config has: distributed.world_size: -1  |  distributed.gpu_ids: []${NC}"
else
    echo -e "${BLUE}GPU mode           :${NC} ${CYAN}SINGLE GPU${NC}"
    echo -e "${BLUE}CUDA_VISIBLE_DEVICES:${NC} ${GPU_IDS}"
    echo -e "${YELLOW}  ↳ Ensure config has: distributed.world_size: 1   |  distributed.gpu_ids: [0]${NC}"
fi
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"

info "Key config fields (read from YAML at runtime):"
echo "    data.dataset_path        — unlabeled HDF5 dataset       (CHANGE ME)"
echo "    logging.checkpoint_dir   — output directory             (CHANGE ME)"
echo "    distributed.world_size   — number of GPUs (-1=all, 1=single)"
echo "    distributed.gpu_ids      — specific GPU indices ([] = all visible)"
echo ""

# --- Run training ------------------------------------------------------------
# Train a single config; returns 0 on success, non-zero on failure (set -e safe).
run_one_config() {
    local cfg="$1"
    echo ""
    info "Starting SSL pre-training: ${cfg}"
    echo ""
    local rc=0
    bioairmet-train --config_path "${cfg}" --mode ssl || rc=$?
    return "${rc}"
}

if [ "${BATCH_MODE}" = true ]; then
    info "Batch mode: ${#CONFIG_FILES[@]} config(s) — one failure will NOT stop the rest."
    SUCCEEDED=0; FAILED=0; FAILED_LIST=()
    for _cfg in "${CONFIG_FILES[@]}"; do
        if run_one_config "${_cfg}"; then
            SUCCEEDED=$((SUCCEEDED + 1)); ok "Succeeded: ${_cfg}"
        else
            FAILED=$((FAILED + 1)); FAILED_LIST+=("${_cfg}"); warn "Failed: ${_cfg}"
        fi
    done
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}Batch summary${NC}: ${GREEN}${SUCCEEDED} succeeded${NC}, ${RED}${FAILED} failed${NC} (of ${#CONFIG_FILES[@]})"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════════${NC}"
    if [ "${FAILED}" -gt 0 ]; then
        echo -e "${RED}Failed configs:${NC}"
        for _f in "${FAILED_LIST[@]}"; do echo -e "  • ${_f}"; done
        echo ""
        error "Batch finished with ${FAILED} failure(s)."
    fi
else
    run_one_config "${CONFIG_FILES[0]}"
fi

# --- Done --------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}✓ SSL pre-training completed successfully!${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Outputs saved to:${NC}  logging.checkpoint_dir/<timestamp>_<project_name>/"
echo -e "  • best_ssl_model.pth   — best validation loss checkpoint"
echo -e "  • last_ssl_model.pth   — final epoch checkpoint"
echo -e "  • config.yaml          — exact config used for this run"
echo ""
echo -e "${CYAN}Next step → Stage 2 (Classification):${NC}"
echo -e "  1. Open  src/bioairmet/config/classification/config_general.yaml"
echo -e "  2. Set   model_initialization.pretraining.experiment_path  to the"
echo -e "           experiment directory printed above"
echo -e "  3. Run:  bash run_classification_training.sh"
echo ""
