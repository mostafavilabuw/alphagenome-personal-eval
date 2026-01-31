#!/bin/bash
################################################################################
# Run PrediXcan Analysis
#
# This script trains ElasticNet models to predict gene expression or chromatin
# accessibility from genetic variants (cis-eQTL/QTL approach).
#
# Usage:
#   bash scripts/examples/run_predixcan.sh
#   bash scripts/examples/run_predixcan.sh --n_regions 200  # Override parameters
#
# Output:
#   - Model performance metrics (CSV)
#   - Trained ElasticNet models (joblib)
#   - Visualization plots (PNG)
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

# Configuration - modify this to switch between LCL and ROSMAP
CONFIG_FILE="configs/lcl_predixcan.yaml"
OUTPUT_BASE_DIR="results/lcl_predixcan"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TIMESTAMP}"

# Print banner
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PrediXcan Analysis - ElasticNet Training${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Project Root : ${PROJECT_ROOT}"
echo "Config       : ${CONFIG_FILE}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "Timestamp    : ${TIMESTAMP}"
echo "Method       : ElasticNet (cis-QTL approach)"
echo ""

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
CMD="python scripts/run_predixcan.py \
    --config ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --save_models true \
    --save_plots true \
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

# Run PrediXcan workflow
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Starting PrediXcan Training...${NC}"
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
    echo -e "${GREEN}  PrediXcan Training Complete!${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo "Elapsed Time : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo "Output Dir   : ${OUTPUT_DIR}"
    echo "Results      : ${OUTPUT_DIR}/predixcan_results.csv"
    echo "Models       : ${OUTPUT_DIR}/models/"
    echo "Plots        : ${OUTPUT_DIR}/figures/"
    echo "Log File     : ${LOG_FILE}"
    echo ""
    exit 0
else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo -e "${RED}================================================================${NC}"
    echo -e "${RED}  PrediXcan Training Failed${NC}"
    echo -e "${RED}================================================================${NC}"
    echo "Elapsed Time : ${ELAPSED}s"
    echo "Log File     : ${LOG_FILE}"
    echo ""
    exit 1
fi
