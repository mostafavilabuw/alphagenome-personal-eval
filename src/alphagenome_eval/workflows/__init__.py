"""
Workflows for AlphaGenome Evaluation

This package provides high-level workflows that combine utility functions
into complete analysis pipelines:

- inference: AlphaGenome predictions with correlation analysis
- predixcan: Variant-based expression prediction with ElasticNet
- atac_predixcan: ATAC-based expression prediction with ElasticNet
- rna_predixcan: RNA-based expression prediction with ElasticNet

Each workflow is designed as a simple function that takes a config dictionary
and returns results as DataFrames for easy downstream analysis.
"""

from .inference import run_inference_workflow
from .predixcan import run_predixcan_workflow
from .atac_predixcan import run_atac_predixcan_workflow, run_atac_predixcan_multi_window
from .rna_predixcan import run_rna_predixcan_workflow

__all__ = [
    'run_inference_workflow',
    'run_predixcan_workflow',
    'run_atac_predixcan_workflow',
    'run_atac_predixcan_multi_window',
    'run_rna_predixcan_workflow',
]
