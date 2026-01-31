# AlphaGenome Personal Evaluation

Evaluation framework for personal genome predictions comparing deep learning models (AlphaGenome, Borzoi) against statistical methods (PrediXcan) for gene expression and chromatin accessibility prediction.

## Citation

If you use this code, please cite:

> **A modality gap in personal-genome prediction by sequence-to-function model**
>
> Xinming Tu, Anna Spiro, Maria Chikina, and Sara Mostafavi
>
> bioRxiv (2026) | [Preprint](https://doi.org/10.1101/XXXX.XXXX)

## Installation

```bash
# Clone the repository
git clone https://github.com/mostafavilabuw/alphagenome-personal-eval.git
cd alphagenome-personal-eval

# Install core dependencies
pip install -e .

# With GPU support (for local Borzoi inference)
pip install -e ".[gpu]"

# All dependencies
pip install -e ".[all]"
```

## Quick Start

### 1. Set up API key

```bash
cp .env.example .env
# Edit .env and add your AlphaGenome API key
```

### 2. Configure data paths

Edit the config files in `configs/` to point to your data:
- `lcl_inference.yaml` - LCL ATAC-seq inference
- `rosmap_inference.yaml` - ROSMAP gene expression inference
- `lcl_predixcan.yaml` - LCL PrediXcan baseline
- `rosmap_predixcan.yaml` - ROSMAP PrediXcan baseline

### 3. Run inference

```bash
# AlphaGenome inference on LCL ATAC-seq
python scripts/run_inference.py --config configs/lcl_inference.yaml

# AlphaGenome inference on ROSMAP gene expression
python scripts/run_inference.py --config configs/rosmap_inference.yaml

# PrediXcan baseline
python scripts/run_predixcan.py --config configs/lcl_predixcan.yaml
```

## Project Structure

```
alphagenome-personal-eval/
├── src/alphagenome_eval/     # Core Python package
│   ├── utils/                # Utility functions
│   │   ├── prediction.py     # AlphaGenome API wrapper
│   │   ├── borzoi_utils.py   # Borzoi/Flashzoi utilities
│   │   ├── visualization.py  # Plotting functions
│   │   ├── modeling.py       # ElasticNet training
│   │   └── data.py           # Data loading
│   ├── workflows/            # Complete pipelines
│   │   ├── inference.py      # Unified inference
│   │   └── predixcan.py      # PrediXcan workflow
│   └── PersonalDataset.py    # Personal genome handling
├── scripts/                  # CLI entry points
├── configs/                  # Configuration files
└── tests/                    # Unit tests
```

## Configuration

Configuration files use YAML format with environment variable support:

```yaml
api_key: ${ALPHAGENOME_API_KEY}
vcf_file_path: ${DATA_DIR}/samples.vcf.gz
```

CLI arguments override config file values:

```bash
python scripts/run_inference.py --config configs/lcl_inference.yaml \
    --n_regions 50 --selection_method random
```

## Data Availability

- **LCL ATAC-seq**: Available from [source]
- **ROSMAP**: Available from Synapse (syn3219045) with data use agreement

## Requirements

- Python 3.9+
- See `pyproject.toml` for full dependency list

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code, please open an issue on GitHub.
