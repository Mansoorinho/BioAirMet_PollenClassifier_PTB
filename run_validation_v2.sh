#!/bin/bash
# =============================================================================
# run_validation_v2.sh — Validate a trained classification model
#
# Runs comprehensive validation on a classification experiment directory,
# producing accuracy metrics, confusion matrices and calibration plots.
#
# Usage:
#   bash run_validation_v2.sh                    # uses paths set below
#   ENV_PATH=/path/to/env bash run_validation_v2.sh
#
# Before running, update the CONFIGURATION block below.
# =============================================================================

set -e

# --- Colour helpers ----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[1;34m'; CYAN='\033[1;36m'; MAGENTA='\033[1;35m'
BOLD='\033[1m'; NC='\033[0m'
ok()      { echo -e "${GREEN}✓${NC} $1"; }
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# --- Banner ------------------------------------------------------------------
cat <<'EOF'

 ██████╗ ██╗ ██████╗      █████╗ ██╗██████╗ ███╗   ███╗███████╗████████╗
 ██╔══██╗██║██╔═══██╗    ██╔══██╗██║██╔══██╗████╗ ████║██╔════╝╚══██╔══╝
 ██████╔╝██║██║   ██║    ███████║██║██████╔╝██╔████╔██║█████╗     ██║   
 ██╔══██╗██║██║   ██║    ██╔══██║██║██╔══██╗██║╚██╔╝██║██╔══╝     ██║   
 ██████╔╝██║╚██████╔╝    ██║  ██║██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   
 ╚═════╝ ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   

           Classification Model Validation (V2 Enhanced)

EOF
echo -e "${BOLD}${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}   BioAirMet V2 Validation - PTB, NPL                              ${NC}"
echo -e "${BOLD}${MAGENTA}════════════════════════════════════════════════════════════════════${NC}\n"

# =============================================================================
# CONFIGURATION — update these fields before running
# =============================================================================

# Path to the classification experiment directory (contains config.yaml + checkpoint)
CHECKPOINT_DIR="/path/to/your/classification_experiment"   # CHANGE ME

# Path to the validation dataset (HDF5 file)
DATA_PATH="/path/to/your/validation_data.h5"               # CHANGE ME

# Which checkpoint to use: "best" (highest val accuracy) or "last" (final epoch)
CHECKPOINT_TYPE="best"

# GPU to use for inference (-1 for CPU)
GPU_ID=0

# Optional: override batch size from config (leave empty to use config value)
BATCH_SIZE=""

# Where to save plots, confusion matrices and the results log
# Default: a timestamped subfolder inside the experiment dir
PLOT_PATH="${CHECKPOINT_DIR}/validation_results"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

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
[ -d "$CHECKPOINT_DIR" ] || error "Experiment directory not found: $CHECKPOINT_DIR"
[ -f "$DATA_PATH" ]      || error "Dataset not found: $DATA_PATH"
[ -f "${CHECKPOINT_DIR}/config.yaml" ] || \
    error "config.yaml not found in: $CHECKPOINT_DIR"

# --- Summary -----------------------------------------------------------------
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}Configuration Summary${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Experiment dir    :${NC} ${CHECKPOINT_DIR}"
echo -e "${BLUE}Checkpoint type   :${NC} ${CHECKPOINT_TYPE}"
echo -e "${BLUE}Dataset           :${NC} ${DATA_PATH}"
echo -e "${BLUE}Output directory  :${NC} ${PLOT_PATH}"
echo -e "${BLUE}GPU ID            :${NC} ${GPU_ID}"
if [ -n "$BATCH_SIZE" ]; then
    echo -e "${BLUE}Batch size        :${NC} ${BATCH_SIZE} (override)"
else
    echo -e "${BLUE}Batch size        :${NC} (from config)"
fi
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"

# --- Run validation ----------------------------------------------------------
info "Starting V2 validation..."
echo ""

BATCH_ARG=""
[ -n "$BATCH_SIZE" ] && BATCH_ARG="--batch_size ${BATCH_SIZE}"

bioairmet-validate \
    --experiment_dir "${CHECKPOINT_DIR}" \
    --checkpoint_type "${CHECKPOINT_TYPE}" \
    --data_path "${DATA_PATH}" \
    --save_path "${PLOT_PATH}" \
    --gpu_id "${GPU_ID}" \
    ${BATCH_ARG}

# --- Done --------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}✓ Validation completed successfully!${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo ""
success "Results saved to: ${PLOT_PATH}"
echo ""
echo -e "${CYAN}Generated outputs:${NC}"
echo "  • validation_log.txt           — detailed results log (Top-1, loss, ACE calibration)"
echo "  • absolute_values_absolute.png — absolute confusion matrix"
echo "  • normalized_values_normalized.png — normalised confusion matrix"
echo ""
