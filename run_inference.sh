#!/usr/bin/env bash
# =============================================================================
# run_inference.sh
#
# Run BioAirMet classification inference on unlabeled raw event data
# OR evaluate on a labeled HDF5 file (with accuracy / confusion matrix).
#
# Image normalisation, batch size, and num_workers are read automatically
# from EXPERIMENT_DIR/config.yaml — no manual configuration needed.
#
# Edit the "User Configuration" block below before running.
# =============================================================================

set -e  # Exit on error

# --- Color Definitions -------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
MAGENTA='\033[1;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Banner ------------------------------------------------------------------
cat <<'EOF'

 ██████╗ ██╗ ██████╗      █████╗ ██╗██████╗ ███╗   ███╗███████╗████████╗
 ██╔══██╗██║██╔═══██╗    ██╔══██╗██║██╔══██╗████╗ ████║██╔════╝╚══██╔══╝
 ██████╔╝██║██║   ██║    ███████║██║██████╔╝██╔████╔██║█████╗     ██║   
 ██╔══██╗██║██║   ██║    ██╔══██║██║██╔══██╗██║╚██╔╝██║██╔══╝     ██║   
 ██████╔╝██║╚██████╔╝    ██║  ██║██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   
 ╚═════╝ ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   

              Classification Inference  (unlabeled or labeled)

EOF

echo -e "${BOLD}${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}   BioAirMet Inference - PTB, NPL                                   ${NC}"
echo -e "${BOLD}${MAGENTA}════════════════════════════════════════════════════════════════════${NC}\n"

# --- Helper Functions --------------------------------------------------------
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Environment Setup -------------------------------------------------------
# Set ENV_PATH to your venv, or leave empty to auto-detect (.venv/venv/env in
# project root), or just activate your environment before running this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$VIRTUAL_ENV" ]; then
    info "Environment already active: ${VIRTUAL_ENV}"
elif [ -n "${ENV_PATH:-}" ] && [ -f "${ENV_PATH}/bin/activate" ]; then
    info "Activating: ${ENV_PATH}"
    source "${ENV_PATH}/bin/activate"
    success "Environment activated: ${VIRTUAL_ENV}"
else
    for _candidate in "${SCRIPT_DIR}/.venv" "${SCRIPT_DIR}/venv" "${SCRIPT_DIR}/env"; do
        if [ -f "${_candidate}/bin/activate" ]; then
            source "${_candidate}/bin/activate"
            success "Environment activated: ${VIRTUAL_ENV}"
            break
        fi
    done
    if [ -z "$VIRTUAL_ENV" ]; then
        warn "No virtual environment found. Using system Python."
        warn "Set ENV_PATH=/path/to/your/env or activate before running."
    fi
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
info "PYTHONPATH includes: ${SCRIPT_DIR}/src"

# =============================================================================
# USER CONFIGURATION — edit these before running
# =============================================================================

# Directory that contains the trained classification model checkpoint
# and its config.yaml (produced by the classification trainer).
EXPERIMENT_DIR=/path/to/your/classification_experiment  # CHANGE ME

# Which checkpoint to load: "best" or "last"
CHECKPOINT_TYPE="best"

# GPU device ID (-1 for CPU)
GPU_ID=0

# Path to data:
#   • RAW EVENT DIRECTORY  → unlabeled inference, predictions CSV is produced
#   • HDF5 FILE            → labeled evaluation, accuracy + confusion matrix produced
DATA_PATH="/path/to/your/data"  # CHANGE ME

# Directory where predictions CSV, logs and plots are saved
SAVE_PATH="./inference_results"  # CHANGE ME

# Confidence threshold for assigning a class label (0.0 = disabled, always use top-1).
# Samples whose top-1 probability is below this value will be labelled "low_confidence".
# Example: 0.75 means only predictions with >= 75% confidence get a class assigned.
CONFIDENCE_THRESHOLD="0.0"

# Predictions CSV detail level:
#   compact  — image_path, predicted_class, probability (slim, production-friendly)
#   detailed — compact + predicted_class_id, top-2/top-3 alternatives and the full
#              per-class probability distribution (prob_<class> columns)
OUTPUT_MODE="compact"

# NOTE: img_reader_type, batch_size, and num_workers are read automatically
#       from ${EXPERIMENT_DIR}/config.yaml — no need to set them here.

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================

# --- Validation Checks -------------------------------------------------------
if [ ! -d "$EXPERIMENT_DIR" ]; then
    error "Experiment directory not found: $EXPERIMENT_DIR"
fi

if [ ! -e "$DATA_PATH" ]; then
    error "Data path not found: $DATA_PATH"
fi

CONFIG_FILE="${EXPERIMENT_DIR}/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    error "config.yaml not found in experiment directory: $EXPERIMENT_DIR"
fi

# --- Configuration Summary ---------------------------------------------------
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}Configuration Summary${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Experiment Directory :${NC} ${EXPERIMENT_DIR}"
echo -e "${BLUE}Checkpoint type      :${NC} ${CHECKPOINT_TYPE}"
echo -e "${BLUE}Config               :${NC} ${CONFIG_FILE}"
echo -e "${BLUE}GPU ID               :${NC} ${GPU_ID}"
echo -e "${BLUE}Data Path            :${NC} ${DATA_PATH}"
echo -e "${BLUE}Save Path            :${NC} ${SAVE_PATH}"
echo -e "${BLUE}Confidence Threshold :${NC} ${CONFIDENCE_THRESHOLD}"
echo -e "${BLUE}img_reader / batch / workers :${NC} read from config.yaml"

# Detect mode
if [ -d "$DATA_PATH" ]; then
    echo -e "${BLUE}Mode                 :${NC} ${YELLOW}UNLABELED${NC} (raw event directory → predictions CSV)"
else
    echo -e "${BLUE}Mode                 :${NC} ${GREEN}LABELED${NC} (HDF5 → accuracy + confusion matrix)"
fi
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"

# --- Run Inference -----------------------------------------------------------
info "Starting inference..."
echo ""

bioairmet-inference \
    --experiment_dir      "${EXPERIMENT_DIR}"      \
    --checkpoint_type     "${CHECKPOINT_TYPE}"     \
    --data_path           "${DATA_PATH}"           \
    --save_path           "${SAVE_PATH}"           \
    --gpu_id              ${GPU_ID}                \
    --confidence_threshold ${CONFIDENCE_THRESHOLD} \
    --output_mode         "${OUTPUT_MODE}"

# --- Post-inference ----------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}✓ Inference completed successfully!${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo ""
success "Results saved to: ${SAVE_PATH}"
echo ""
echo -e "${CYAN}Generated files:${NC}"
echo -e "  • predictions_<timestamp>.csv    - Per-sample predictions and probabilities (detail level: ${OUTPUT_MODE})"
echo -e "  • inference_summary_<timestamp>.json - Machine-readable run summary"
echo -e "  • inference_<timestamp>.log      - Detailed run log"
echo -e "  • absolute_values_absolute.png     - (labeled mode only)"
echo -e "  • normalized_values_normalized.png - (labeled mode only)"
echo ""

