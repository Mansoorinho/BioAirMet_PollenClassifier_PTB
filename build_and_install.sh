#!/bin/bash
# =============================================================================
# BioAirMet - Build & Install Script
#
# Installs the BioAirMet package (editable) + all dependencies into a Python
# environment, then verifies the installation and the CLI entry points.
#
# Two modes:
#   • CREATE  (recommended) — create a FRESH virtual environment (clean,
#     isolated, no dependency conflicts with other projects).
#   • REUSE   — install into an environment you already have (your choice).
#
# Usage:
#   bash build_and_install.sh                          # interactive (asked)
#   bash build_and_install.sh <env_path> <python_bin>  # non-interactive, new env
#   bash build_and_install.sh --reuse-env              # reuse active/ENV_PATH env
#   bash build_and_install.sh --reuse-env /path/env    # reuse the env at /path/env
#
# Flags:
#   --reuse-env [PATH]   Install into an existing environment instead of
#                        creating one. PATH may be omitted — the script then
#                        falls back to $VIRTUAL_ENV, then $ENV_PATH, then the
#                        environment of the currently active `python`.
#   --python BIN         Python binary to use (default: auto-detected).
#   -h | --help          Show this help and exit.
#
# Environment variables (shared with the run_*.sh launcher scripts):
#   ENV_PATH             Venv path — pre-fills the prompt in interactive mode
#                        and is the fallback target of --reuse-env.
#   BIOMET_TORCH_BUILD   PyTorch build selection (skips the interactive
#                        prompt): auto (default — ask, or pick the newest
#                        build your NVIDIA driver supports), driver,
#                        latest (default PyPI build), cpu, or an explicit
#                        wheel channel such as cu118, cu124, cu126, cu128.
#
# Defaults:
#   Env path   : ${ENV_PATH:-<project_dir>/env}
#   Python bin : auto-detected (first of: python3.11, python3.10, python3)
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Colours -----------------------------------------------------------------
GREEN='\033[0;32m'; CYAN='\033[1;36m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${CYAN}▶${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
fail() { echo -e "${RED}✗ ERROR:${NC} $1"; exit 1; }

# --- Help (prints the header usage block) -------------------------------------
usage() {
    sed -n '2,39p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# --- Banner ------------------------------------------------------------------
echo ""
echo -e "${BOLD}=========================================="
echo -e " BioAirMet - Build & Install"
echo -e "==========================================${NC}"
echo ""

# --- Detect default Python binary --------------------------------------------
DEFAULT_PYTHON_BIN=""
for _py in python3.11 python3.10 python3; do
    if command -v "${_py}" &>/dev/null; then
        DEFAULT_PYTHON_BIN="$(command -v "${_py}")"
        break
    fi
done
[ -z "$DEFAULT_PYTHON_BIN" ] && DEFAULT_PYTHON_BIN="/usr/bin/python3"

# ENV_PATH is the shared environment variable — the run_*.sh launcher scripts
# all honour it too, so a single export covers every script.
DEFAULT_ENV_PATH="${ENV_PATH:-${PROJECT_DIR}/env}"

# =============================================================================
# CLI parsing
# =============================================================================
MODE="new"          # "new" (create fresh venv) or "reuse" (existing env)
REUSE_PATH=""
ARG_PYTHON=""
POS_ENV=""
POS_PY=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --reuse-env)
            MODE="reuse"
            shift
            # Optional explicit path: the next argument, if it is not a flag
            if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
                REUSE_PATH="$1"; shift
            fi
            ;;
        --python)
            [ $# -ge 2 ] || fail "--python requires a Python binary path (see --help)"
            ARG_PYTHON="$2"; shift 2
            ;;
        --*)
            fail "Unknown option: $1 (see --help)"
            ;;
        *)
            # Positional (legacy order): <env_path> [python_bin]
            if [ -z "$POS_ENV" ]; then
                POS_ENV="$1"
            elif [ -z "$POS_PY" ]; then
                POS_PY="$1"
            else
                fail "Too many positional arguments (expected: <env_path> [python_bin])"
            fi
            shift
            ;;
    esac
done

# --- Resolve target environment + Python binary ------------------------------
if [ "$MODE" = "reuse" ]; then
    # --reuse-env resolution order:
    #   1) explicit path from --reuse-env
    #   2) active virtualenv ($VIRTUAL_ENV)
    #   3) $ENV_PATH (shared launcher-script variable)
    #   4) the environment of the currently active `python`
    TARGET="${REUSE_PATH:-${VIRTUAL_ENV:-${ENV_PATH:-}}}"
    if [ -z "$TARGET" ]; then
        if command -v python &>/dev/null; then
            TARGET="$(cd "$(dirname "$(command -v python)")/.." && pwd)"
        else
            fail "No environment found to reuse.
  Activate your venv first, or pass one explicitly:
    bash build_and_install.sh --reuse-env /path/to/env"
        fi
    fi
    ENV_PATH="$TARGET"
    PYTHON_BIN="${ARG_PYTHON:-${ENV_PATH}/bin/python}"
    warn "Reusing the existing environment: ${ENV_PATH}"
    warn "A FRESH virtual environment is recommended (clean, isolated, no"
    warn "dependency conflicts). Create one instead:"
    warn "    bash build_and_install.sh /path/to/new_env /usr/bin/python3.11"
else
    if [ -n "$POS_ENV" ]; then
        # Non-interactive: positional argument(s)
        ENV_PATH="${POS_ENV}"
        PYTHON_BIN="${POS_PY:-${ARG_PYTHON:-${DEFAULT_PYTHON_BIN}}}"
        if [ -z "$POS_PY" ] && [ -z "$ARG_PYTHON" ]; then
            warn "No Python binary given — using default: ${PYTHON_BIN}"
        fi
    else
        # --- Interactive ------------------------------------------------------
        if [ -n "${VIRTUAL_ENV:-}" ]; then
            echo -e "${CYAN}How would you like to set up the environment?${NC}"
            echo -e "  1) ${BOLD}Create a FRESH virtual environment (recommended)${NC}"
            echo -e "  2) Reuse the ${BOLD}active${NC} environment: ${YELLOW}${VIRTUAL_ENV}${NC}"
            echo -e "  3) Reuse an existing environment at a ${BOLD}path${NC} of your choice"
        else
            echo -e "${CYAN}How would you like to set up the environment?${NC}"
            echo -e "  1) ${BOLD}Create a FRESH virtual environment (recommended)${NC}"
            echo -e "  2) Reuse an existing environment at a ${BOLD}path${NC} of your choice"
        fi
        read -r -p "Choose an option [1]: " _choice
        _choice="${_choice:-1}"

        case "$_choice" in
            2)
                if [ -n "${VIRTUAL_ENV:-}" ]; then
                    ENV_PATH="${VIRTUAL_ENV}"
                    PYTHON_BIN="${ENV_PATH}/bin/python"
                else
                    read -r -p "  Enter existing environment path: " _re
                    [ -n "${_re}" ] || fail "No path given for reuse."
                    ENV_PATH="${_re}"
                    PYTHON_BIN="${ENV_PATH}/bin/python"
                fi
                warn "Reusing an existing environment — a fresh one is recommended."
                ;;
            3)
                read -r -p "  Enter existing environment path: " _re
                [ -n "${_re}" ] || fail "No path given for reuse."
                ENV_PATH="${_re}"
                PYTHON_BIN="${ENV_PATH}/bin/python"
                warn "Reusing an existing environment — a fresh one is recommended."
                ;;
            *)
                echo -e "${CYAN}Please provide the following paths (press Enter to accept the default):${NC}"
                echo ""
                echo -e "  ${BOLD}Python binary${NC}"
                echo -e "  Default: ${YELLOW}${DEFAULT_PYTHON_BIN}${NC}"
                read -r -p "  Enter Python binary path [${DEFAULT_PYTHON_BIN}]: " _input_py
                PYTHON_BIN="${_input_py:-${ARG_PYTHON:-${DEFAULT_PYTHON_BIN}}}"
                echo ""
                echo -e "  ${BOLD}Virtual environment path${NC}"
                echo -e "  Default: ${YELLOW}${DEFAULT_ENV_PATH}${NC}"
                read -r -p "  Enter environment path [${DEFAULT_ENV_PATH}]: " _input_env
                ENV_PATH="${_input_env:-${DEFAULT_ENV_PATH}}"
                echo ""
                ;;
        esac
    fi
fi

# Expand ~ if user typed it
ENV_PATH="${ENV_PATH/#\~/$HOME}"
PYTHON_BIN="${PYTHON_BIN/#\~/$HOME}"

echo -e "${BOLD}  Project : ${NC}${PROJECT_DIR}"
echo -e "${BOLD}  Env     : ${NC}${ENV_PATH}"
echo -e "${BOLD}  Python  : ${NC}${PYTHON_BIN}"
echo ""

# --- Sanity checks -----------------------------------------------------------
[ -f "${PROJECT_DIR}/pyproject.toml" ] || \
    fail "pyproject.toml not found. Run this script from the project root."

if [ "$MODE" = "reuse" ]; then
    [ -f "${ENV_PATH}/bin/activate" ] || \
        fail "Not a virtual environment (no bin/activate): ${ENV_PATH}
  To create a fresh one instead, run:
    bash build_and_install.sh /path/to/env /usr/bin/python3.11"
fi

[ -x "${PYTHON_BIN}" ] || \
    fail "Python binary not found or not executable: ${PYTHON_BIN}
  Hint: pass one explicitly, e.g.
    bash build_and_install.sh --reuse-env ${ENV_PATH} --python /usr/bin/python3.11"

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
info "Python version: ${PYTHON_VERSION}"

# Require Python 3.10+
"${PYTHON_BIN}" -c '
import sys
if sys.version_info < (3, 10):
    print("ERROR: Python 3.10+ is required (bioairmet requires-python >= 3.10).")
    sys.exit(1)
' || fail "Python 3.10+ is required."

# --- Step 1: Virtual environment ---------------------------------------------
echo ""
echo -e "${BOLD}Step 1: Virtual environment${NC}"

if [ "$MODE" = "reuse" ]; then
    ok "Reusing existing environment at ${ENV_PATH} (as requested)."
elif [ -d "${ENV_PATH}" ] && [ -f "${ENV_PATH}/bin/activate" ]; then
    warn "Environment already exists at ${ENV_PATH} — skipping creation."
else
    info "Creating virtual environment at: ${ENV_PATH}"
    mkdir -p "$(dirname "${ENV_PATH}")"
    "${PYTHON_BIN}" -m venv "${ENV_PATH}"
    ok "Virtual environment created."
fi

# Activate for the remainder of this script
source "${ENV_PATH}/bin/activate"
ok "Activated: ${VIRTUAL_ENV}"

ENV_PYTHON="${ENV_PATH}/bin/python"
ENV_PIP="${ENV_PATH}/bin/pip"

# --- Step 2: Upgrade pip + install build tools -------------------------------
echo ""
echo -e "${BOLD}Step 2: Upgrade pip & install build tools${NC}"

info "Upgrading pip, setuptools, wheel..."
"${ENV_PIP}" install --quiet --upgrade pip setuptools wheel
ok "pip $("${ENV_PIP}" --version | awk '{print $2}') ready."

"${ENV_PIP}" install --quiet build
ok "build frontend ready."

# --- Step 3: Select the PyTorch build (CUDA driver check) --------------------
echo ""
echo -e "${BOLD}Step 3: Select PyTorch build (CUDA driver check)${NC}"

# --- Detect the NVIDIA driver ------------------------------------------------
# The nvidia-smi banner reports the newest CUDA runtime the installed driver
# can run ("CUDA Version: X.Y"). A PyTorch wheel built for cuXZ works iff
# Z <= Y (CUDA minor versions are forward-compatible within one major).
# Every probe is guarded so the script never aborts under `set -e`.
NVIDIA_GPU_NAME=""
NVIDIA_DRIVER=""
DRIVER_CUDA_MAX=""
if command -v nvidia-smi &>/dev/null; then
    if _smi="$(nvidia-smi 2>/dev/null)" && [ -n "$_smi" ]; then
        DRIVER_CUDA_MAX="$(printf '%s\n' "$_smi" | grep -oE 'CUDA Version: [0-9.]+' | awk '{print $3}' | head -1)"
        NVIDIA_DRIVER="$(printf '%s\n' "$_smi" | grep -oE 'Driver Version: [0-9.]+' | awk '{print $3}' | head -1)"
        NVIDIA_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    fi
fi

if [ -n "$NVIDIA_GPU_NAME" ]; then
    info "GPU           : ${NVIDIA_GPU_NAME}"
    info "Driver        : ${NVIDIA_DRIVER:-unknown}"
    info "Driver CUDA   : supports up to CUDA ${DRIVER_CUDA_MAX:-unknown}"
else
    info "GPU           : no NVIDIA GPU detected"
fi

# --- Map the driver's CUDA support to the newest PyTorch channel it can run ---
# (channels verified to exist on download.pytorch.org, 2026-08)
recommend_channel() {
    case "$1" in
        13.*)          echo "cu130" ;;
        12.9)          echo "cu129" ;;
        12.8)          echo "cu128" ;;
        12.7|12.6)     echo "cu126" ;;
        12.5|12.4)     echo "cu124" ;;
        *)             echo "cu118" ;;   # CUDA 11.x and CUDA 12.0 - 12.3
    esac
}

# --- Resolve the user's decision ---------------------------------------------
# BIOMET_TORCH_BUILD: auto (default) | driver | latest | cpu | cu<XXX>
TORCH_CHOICE="${BIOMET_TORCH_BUILD:-auto}"
case "$TORCH_CHOICE" in
    auto|driver|latest|cpu|cu*) : ;;
    *) fail "Invalid BIOMET_TORCH_BUILD='${TORCH_CHOICE}'. Expected: auto, driver, latest, cpu, or an explicit channel (cu118, cu124, cu126, cu128, ...)." ;;
esac

if [ -n "$NVIDIA_GPU_NAME" ] && [ -n "$DRIVER_CUDA_MAX" ]; then
    RECOMMENDED_CHANNEL="$(recommend_channel "$DRIVER_CUDA_MAX")"
elif [ -n "$NVIDIA_GPU_NAME" ]; then
    # GPU present but the driver's CUDA version could not be parsed: fall back
    # to the latest build and let the post-install check diagnose the GPU.
    RECOMMENDED_CHANNEL=""
else
    RECOMMENDED_CHANNEL="cpu"
fi

# The latest default-PyPI build (torch 2.9.x as of 2026-08) bundles CUDA 12.8.
LATEST_CUDA="12.8"
latest_fits_driver() {
    if [ -z "$DRIVER_CUDA_MAX" ]; then
        return 1
    fi
    local drv_num latest_num
    drv_num="$(printf '%s' "$DRIVER_CUDA_MAX" | awk -F. '{print $1*100+$2}')"
    latest_num="$(printf '%s' "$LATEST_CUDA" | awk -F. '{print $1*100+$2}')"
    [ "$drv_num" -ge "$latest_num" ]
}

if [ "$TORCH_CHOICE" = "auto" ]; then
    if [ -n "$NVIDIA_GPU_NAME" ] && [ -n "$RECOMMENDED_CHANNEL" ] && [ -t 0 ]; then
        # ---------------- Interactive prompt ----------------
        OLD_PIN=0
        case "$RECOMMENDED_CHANNEL" in cu118|cu124) OLD_PIN=1 ;; esac
        echo ""
        echo -e "${CYAN}Which PyTorch build should be installed?${NC}"
        echo -e "  1) ${BOLD}Match your NVIDIA driver${NC} (recommended) — ${RECOMMENDED_CHANNEL} wheels (newest PyTorch your driver can run)"
        if [ "$OLD_PIN" = 1 ]; then
            echo -e "      Note: the newest ${RECOMMENDED_CHANNEL} wheel is torch 2.7.x — below the tested minimum (torch>=2.8.0), but it works with your GPU. Updating the NVIDIA driver would allow the latest PyTorch."
        fi
        echo -e "  2) ${BOLD}Latest${NC} — newest default PyPI build (bundles CUDA ${LATEST_CUDA})"
        if ! latest_fits_driver; then
            echo -e "      Note: this build needs a driver supporting CUDA >= ${LATEST_CUDA} — your GPU will NOT be used on this machine."
        fi
        echo -e "  3) ${BOLD}CPU only${NC} — no CUDA libraries"
        read -r -p "Choose an option [1]: " _tchoice
        case "${_tchoice:-1}" in
            2) TORCH_CHOICE="latest" ;;
            3) TORCH_CHOICE="cpu" ;;
            *) TORCH_CHOICE="driver" ;;
        esac
    elif [ -n "$NVIDIA_GPU_NAME" ]; then
        # Non-interactive (CI / piped input): use the recommendation.
        if [ -n "$RECOMMENDED_CHANNEL" ]; then
            TORCH_CHOICE="driver"
        else
            TORCH_CHOICE="latest"
            warn "Could not determine the driver's CUDA version — using the latest build; the post-install check will diagnose any GPU issue."
        fi
        info "Non-interactive — using the '${TORCH_CHOICE}' build (override with BIOMET_TORCH_BUILD=driver|latest|cpu|cuXXX)."
    fi
fi

# The only remaining 'auto' case is a machine without an NVIDIA GPU:
# install the CPU-only build (the CUDA builds would just waste ~2.7 GB).
if [ "$TORCH_CHOICE" = "auto" ]; then
    TORCH_CHOICE="cpu"
    info "No NVIDIA GPU detected — using the CPU-only build (override with BIOMET_TORCH_BUILD=latest|cuXXX)."
fi

# --- Finalise mode + channel ---------------------------------------------------
TORCH_INSTALL_MODE="latest"
TORCH_CHANNEL=""
case "$TORCH_CHOICE" in
    driver)
        if [ -z "$RECOMMENDED_CHANNEL" ]; then
            fail "No NVIDIA GPU (or unreadable driver) — a 'driver' build is not possible. Use BIOMET_TORCH_BUILD=latest|cpu|cuXXX instead."
        fi
        TORCH_INSTALL_MODE="channel"
        TORCH_CHANNEL="$RECOMMENDED_CHANNEL"
        ;;
    latest)
        TORCH_INSTALL_MODE="latest"
        ;;
    cpu)
        TORCH_INSTALL_MODE="channel"
        TORCH_CHANNEL="cpu"
        ;;
    cu*)
        TORCH_INSTALL_MODE="channel"
        TORCH_CHANNEL="$TORCH_CHOICE"
        ;;
esac

if [ "$TORCH_INSTALL_MODE" = "latest" ]; then
    info "PyTorch build : latest (default PyPI build)"
else
    info "PyTorch build : ${TORCH_CHANNEL} channel (newest in channel)"
fi

# --- Step 4: Install the project (editable + all deps) -----------------------
echo ""
echo -e "${BOLD}Step 4: Install BioAirMet + dependencies${NC}"

cd "${PROJECT_DIR}"

# 1) PyTorch trio first — the package pins (torch>=2.8.0, ...) are then
#    satisfied by exactly the build chosen in Step 3.
if [ "$TORCH_INSTALL_MODE" = "latest" ]; then
    info "Installing the newest default PyPI PyTorch build..."
    "${ENV_PIP}" install --upgrade torch torchvision torchaudio
else
    info "Installing the newest PyTorch from the ${TORCH_CHANNEL} channel..."
    # --index-url points at the PyTorch wheel index (a full PyPI mirror plus
    # the CUDA wheels), so the trio resolves to ${TORCH_CHANNEL} builds.
    "${ENV_PIP}" install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
fi

# 2) BioAirMet (editable) + the remaining dependencies.
info "Installing BioAirMet in editable mode..."
TORCH_BELOW_MIN=""
TORCH_BELOW_MIN="$("${ENV_PYTHON}" -c 'import torch; print("yes" if tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) < (2, 8) else "no")' 2>/dev/null)" || true
if [ "$TORCH_BELOW_MIN" = "yes" ]; then
    warn "The installed torch is below the package minimum (torch>=2.8.0)."
    warn "Installing BioAirMet without its dependency pins, then the remaining"
    warn "dependencies explicitly (torch/torchvision/torchaudio excluded)."
    "${ENV_PIP}" install -e . --no-deps
    "${ENV_PYTHON}" - "${PROJECT_DIR}/pyproject.toml" <<'PY' | "${ENV_PIP}" install -r /dev/stdin
import sys, tomllib

with open(sys.argv[1], "rb") as f:
    deps = tomllib.load(f)["project"]["dependencies"]
for dep in deps:
    if not dep.lower().startswith(("torch", "torchvision", "torchaudio")):
        print(dep)
PY
else
    "${ENV_PIP}" install -e .
fi

ok "BioAirMet installed in editable mode."

# --- Step 5: Verify imports --------------------------------------------------
echo ""
echo -e "${BOLD}Step 5: Verify installation${NC}"

# Passed to the verification snippet below for the CUDA compatibility report.
export DRIVER_CUDA_MAX TORCH_INSTALL_MODE TORCH_CHANNEL

"${ENV_PYTHON}" << 'PYEOF'
import sys

failures = []

checks = [
    # (import_statement, label)
    ("import bioairmet",                                            "bioairmet package"),
    ("from bioairmet import HoloClassifierV2",                      "HoloClassifierV2"),
    ("from bioairmet import SSLTrainerV2, ClassificationTrainerV2", "V2 trainers"),
    ("from bioairmet import parse_config, build_model_from_config", "config + model builders"),
    ("from bioairmet import Stage2Dataset, ValidationDataset_Unlabeled", "datasets"),
    ("from bioairmet.utils import ace",                             "calibration (ace)"),
    ("from bioairmet import models, data, utils, trainers, training", "all submodules"),
    ("from bioairmet.cli import main_train, main_inference, main_validate", "CLI entry points"),
    ("from bioairmet.training.inference import run_inference",   "inference engine"),
    # Third-party deps
    ("import torch",       "torch"),
    ("import torchvision", "torchvision"),
    ("import timm",        "timm"),
    ("import h5py",        "h5py"),
    ("import numpy",       "numpy"),
    ("import pandas",      "pandas"),
    ("import matplotlib",  "matplotlib"),
    ("import seaborn",     "seaborn"),
    ("import sklearn",     "scikit-learn"),
    ("import psutil",      "psutil"),
    ("import tensorboard", "tensorboard"),
    ("import tqdm",        "tqdm"),
    ("import yaml",        "pyyaml"),
    ("import easydict",    "easydict"),
]

col_w = max(len(label) for _, label in checks) + 2

for stmt, label in checks:
    try:
        exec(stmt)
        print(f"  \033[32m✓\033[0m  {label:{col_w}}")
    except Exception as e:
        print(f"  \033[31m✗\033[0m  {label:{col_w}}  ← {e}")
        failures.append(label)

import bioairmet
print(f"\n  Version : {bioairmet.__version__}")
import os
import torch
print(f"  PyTorch : {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA    : {torch.version.cuda}  |  GPUs: {torch.cuda.device_count()}")
else:
    drv = (os.environ.get("DRIVER_CUDA_MAX") or "").strip()
    if drv:
        print(f"  CUDA    : NOT AVAILABLE — this torch build targets CUDA "
              f"{torch.version.cuda or '(CPU build)'} but the driver supports up to {drv}")
        print("            Fix: re-run build_and_install.sh and choose 'match your NVIDIA driver',")
        print("            update the NVIDIA driver, or run on CPU (gpu: -1 in the config).")
    else:
        print("  CUDA    : no NVIDIA GPU detected — CPU-only operation is expected.")

if failures:
    print(f"\n\033[31m✗ {len(failures)} check(s) failed: {', '.join(failures)}\033[0m")
    sys.exit(1)
else:
    print("\n\033[32m✓ All checks passed!\033[0m")
PYEOF

# --- Step 6: Verify CLI commands ---------------------------------------------
echo ""
echo -e "${BOLD}Step 6: Verify CLI commands${NC}"

for cmd in bioairmet-train bioairmet-inference bioairmet-validate; do
    if "${ENV_PATH}/bin/${cmd}" --help > /dev/null 2>&1; then
        ok "${cmd}"
    else
        warn "${cmd} — not found or failed (reinstall with: pip install -e .)"
    fi
done

# --- Done --------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}=========================================="
echo -e " Installation complete!"
echo -e "==========================================${NC}"
echo ""
echo -e "  ${CYAN}Environment activated:${NC} ${ENV_PATH}"
echo ""
echo -e "  To activate in a new shell:"
echo -e "    ${YELLOW}source ${ENV_PATH}/bin/activate${NC}"
echo ""
echo -e "  Or, without activating (the run_*.sh scripts all honour the same variable):"
echo -e "    ${YELLOW}export ENV_PATH=${ENV_PATH}${NC}"
echo ""
echo -e "  Quick start:"
echo "    python -c 'import bioairmet; print(bioairmet.__version__)'"
echo "    bioairmet-train --help"
echo ""
echo "  See README.md for the full usage guide."
echo ""

# Activate the environment in the current shell session
# (If sourced: `. build_and_install.sh`, this will persist in the calling shell.
#  If run with `bash build_and_install.sh`, the environment was active for this
#  script's duration; use the source command printed above to re-activate.)
