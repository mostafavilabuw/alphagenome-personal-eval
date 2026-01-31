#!/bin/bash
################################################################################
# Run ROSMAP Gene Expression Inference with AlphaGenome
#
# This script runs large-scale inference on ROSMAP brain gene expression
# (dorsolateral prefrontal cortex) using the AlphaGenome API.
#
# Usage:
#   bash scripts/examples/run_rosmap_inference.sh
#   bash scripts/examples/run_rosmap_inference.sh --n_regions 100  # Override parameters
#
# Output:
#   - Correlation results (CSV)
#   - Raw predictions (NPZ files)
#   - Run metadata and logs
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}"

# Configuration
CONFIG_FILE="configs/rosmap_inference.yaml"
OUTPUT_BASE_DIR="results/rosmap_inference"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TIMESTAMP}"

# Print banner
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  ROSMAP Gene Expression Inference - AlphaGenome Evaluation${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Project Root : ${PROJECT_ROOT}"
echo "Config       : ${CONFIG_FILE}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "Timestamp    : ${TIMESTAMP}"
echo "Tissue       : Dorsolateral Prefrontal Cortex (DLPFC)"
echo ""

# Check for .env file and API key
if [ ! -f .env ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create a .env file with your ALPHAGENOME_API_KEY."
    echo "Example: cp .env.example .env && vim .env"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

if [ -z "${ALPHAGENOME_API_KEY:-}" ]; then
    echo -e "${RED}ERROR: ALPHAGENOME_API_KEY not set in .env file!${NC}"
    exit 1
fi

echo -e "${GREEN}API key found${NC}"

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}ERROR: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

echo -e "${GREEN}Config file found${NC}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}Output directory created${NC}"
echo ""

# Log file
LOG_FILE="${OUTPUT_DIR}/run.log"

# Build command
CMD="python scripts/run_inference.py \
    --config ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --save_predictions \
    --verbose \
    --log_file ${LOG_FILE}"

# Add any additional arguments passed to this script
if [ $# -gt 0 ]; then
    CMD="${CMD} $*"
    echo -e "${YELLOW}Additional arguments: $*${NC}"
    echo ""
fi

# Print command
echo -e "${BLUE}Running command:${NC}"
echo "${CMD}"
echo ""

# Run inference
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Starting ROSMAP Inference...${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

START_TIME=$(date +%s)

if ${CMD}; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))

    echo ""
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${GREEN}  ROSMAP Inference Complete!${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo "Elapsed Time : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo "Output Dir   : ${OUTPUT_DIR}"
    echo "Results      : ${OUTPUT_DIR}/inference_results.csv"
    echo "Log File     : ${LOG_FILE}"
    echo ""
    exit 0
else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo -e "${RED}================================================================${NC}"
    echo -e "${RED}  ROSMAP Inference Failed${NC}"
    echo -e "${RED}================================================================${NC}"
    echo "Elapsed Time : ${ELAPSED}s"
    echo "Log File     : ${LOG_FILE}"
    echo ""
    exit 1
fi
