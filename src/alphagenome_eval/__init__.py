"""
AlphaGenome Evaluation Package

This package provides tools for evaluating AlphaGenome predictions on genomic data.

Key components:
- GenomeDataset: Load personal genomes with variants
- utils: Reusable utility functions for data loading, genome operations,
  predictions, modeling, and visualization
- save_predictions: Functions for saving/loading predictions

Example usage:
    >>> from alphagenome_eval import GenomeDataset
    >>> from alphagenome_eval.utils import load_peaks, extract_variants_in_region
    >>>
    >>> # Load data
    >>> metadata, expr, samples = load_peaks('./data/LCL', n_peaks=100)
    >>>
    >>> # Create dataset
    >>> dataset = GenomeDataset(
    ...     gene_metadata=metadata,
    ...     vcf_file_path='genotypes.vcf.gz',
    ...     hg38_file_path='hg19.fa',
    ...     expr_data=expr,
    ...     sample_list=samples,
    ...     window_size=100000
    ... )
"""

__version__ = "0.1.0"

from .save_predictions import save_alphagenome_predictions, load_alphagenome_predictions

# Personal dataset functionality (requires pysam)
try:
    from .PersonalDataset import GenomeDataset, onehot_encoding
    _PYSAM_AVAILABLE = True
except ImportError:
    _PYSAM_AVAILABLE = False
    import warnings
    warnings.warn(
        "pysam not available. PersonalDataset functionality disabled. "
        "Install pysam to use personal genome features: pip install pysam",
        ImportWarning
    )

# Import utils submodule for convenient access
from . import utils

__all__ = [
    # Prediction I/O
    'save_alphagenome_predictions',
    'load_alphagenome_predictions',
    # Utils module
    'utils',
]

# Add PersonalDataset exports if available
if _PYSAM_AVAILABLE:
    __all__.extend(['GenomeDataset', 'onehot_encoding']) 