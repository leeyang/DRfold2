#!/bin/bash

# ==============================================================================
# RNAref3D Batch Refinement Script
# 
# Description: 
#   Sequentially processes PDB files from an input directory using the 
#   RNAref3D pipeline (Python prediction + C++ refinement) on a single GPU.
#
# Usage: 
#   ./batch_refine.sh <gpu_id> <input_pdb_dir> [output_dir]
# ==============================================================================

# Exit immediately if a command exits with a non-zero status (during setup)
set -e

# --- Configuration ---

# robustly get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="${SCRIPT_DIR}"

# Define paths to executables relative to the script location
PREDICTOR_SCRIPT="${REPO_ROOT}/RNAref3Dbp/main.py"
REFINER_EXE="${REPO_ROOT}/bin/RNAref3D"

# --- Argument Parsing ---

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu_id> <input_pdb_dir> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  gpu_id         : The ID of the GPU to use (e.g., 0)."
    echo "  input_pdb_dir  : Directory containing input .pdb files."
    echo "  output_dir     : (Optional) Directory to save results. Defaults to ./refinement_results."
    echo ""
    exit 1
fi

GPU_ID=$1
INPUT_DIR="$2"
# Use provided output dir or default to 'refinement_results'
BASE_OUTPUT_DIR="${3:-${REPO_ROOT}/refinement_results}"

# --- Pre-flight Checks ---

if [ ! -f "$PREDICTOR_SCRIPT" ]; then
    echo "Error: Python predictor not found at: $PREDICTOR_SCRIPT"
    echo "Please ensure the script is located in the RNAref3D repository root."
    exit 1
fi

if [ ! -f "$REFINER_EXE" ]; then
    echo "Error: C++ executable not found at: $REFINER_EXE"
    echo "Please compile the C++ source code first (check bin/ directory)."
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

echo "=================================================="
echo "RNAref3D Batch Processing Started"
echo "--------------------------------------------------"
echo "GPU ID           : ${GPU_ID}"
echo "Input Directory  : ${INPUT_DIR}"
echo "Output Directory : ${BASE_OUTPUT_DIR}"
echo "=================================================="

# --- Main Processing Loop ---

# Disable 'set -e' to prevent the script from stopping if a single file fails
set +e

# Find all .pdb files (Case insensitive)
# NOTE: Adjusted to *.pdb for general usage.
find "${INPUT_DIR}" -maxdepth 1 -type f -name "*.pdb" | sort | while read pdb_file; do
    
    # Extract filename without extension
    pdb_basename=$(basename "${pdb_file}" .pdb)
    
    # Create specific output folder for this case
    case_output_dir="${BASE_OUTPUT_DIR}/${pdb_basename}"
    mkdir -p "${case_output_dir}"

    echo ""
    echo ">>> Processing: ${pdb_basename}"
    echo "    File: ${pdb_file}"

    # ---------------------------------------------------------
    # Step 1: Base Pair and Stacking Prediction (Python)
    # ---------------------------------------------------------
    echo "    [Step 1/2] Running Python Predictor..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${PREDICTOR_SCRIPT}" \
        --pdb "${pdb_file}" \
        -o "${case_output_dir}" > "${case_output_dir}/step1_predict.log" 2>&1

    # Check if Step 1 produced necessary files
    bp_file="${case_output_dir}/RNAref3D_bp.txt"
    st_file="${case_output_dir}/RNAref3D_stack.txt"

    if [ ! -f "${bp_file}" ] || [ ! -f "${st_file}" ]; then
        echo "    [ERROR] Prediction failed. Missing output files."
        echo "    Check logs at: ${case_output_dir}/step1_predict.log"
        continue # Skip to next file
    fi
    echo "    Step 1 Complete."

    # ---------------------------------------------------------
    # Step 2: Full-Atom Refinement (C++)
    # ---------------------------------------------------------
    echo "    [Step 2/2] Running C++ Refinement..."

    "${REFINER_EXE}" \
        -i "${pdb_file}" \
        -o "${case_output_dir}" \
        -bp "${bp_file}" \
        -st "${st_file}" > "${case_output_dir}/step2_refine.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "    [ERROR] Refinement failed during C++ execution."
        echo "    Check logs at: ${case_output_dir}/step2_refine.log"
        continue
    fi

    echo "    Step 2 Complete. Results saved to: ${case_output_dir}"

done

echo ""
echo "=================================================="
echo "Batch processing finished."
echo "=================================================="