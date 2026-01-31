"""
PrediXcan Workflow for Variant-Based Expression Prediction

This module provides a unified workflow for running PrediXcan-style analysis:
extracting variants, training ElasticNet models, and evaluating predictions.

Consolidates:
- explore/LCL/prediXcan/scripts/run_predixcan_analysis.py
- ROSMAP PrediXcan scripts

Author: AlphaGenome Evaluation Team
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from alphagenome_eval.utils import (
    load_peaks,
    load_genes,
    load_vcf_samples,
    extract_variants_in_region,
    filter_variants_by_maf,
    train_elasticnet,
    evaluate_predictions,
    summarize_predixcan_results,
    create_comprehensive_predixcan_plots
)

logger = logging.getLogger(__name__)


def run_predixcan_workflow(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete PrediXcan workflow for variant-based predictions.

    This function orchestrates the entire PrediXcan pipeline:
    1. Load genomic regions (peaks/genes) and expression data
    2. Extract variants in context windows around each region
    3. Train ElasticNet models to predict expression from genotypes
    4. Evaluate models on train/test (and optionally validation) sets
    5. Generate summary statistics and visualizations

    Args:
        config: Configuration dictionary with keys:
            - data_type: 'peaks' or 'genes'
            - data_dir: Path to data directory (for peaks)
            - gene_meta_path: Path to gene metadata (for genes)
            - expr_data_path: Path to expression data (for genes)
            - vcf_file_path: Path to VCF file
            - context_window: Genomic window size for variant extraction (default: 100000)
            - min_maf: Minimum minor allele frequency (default: 0.05)
            - n_regions: Number of regions to process (optional)
            - test_size: Test set fraction (default: 0.2)
            - val_size: Validation set fraction (default: 0.0, use if > 0)
            - l1_ratio: ElasticNet L1 ratio (default: 0.5)
            - alphas: List of alpha values to try (default: auto)
            - random_seed: Random seed (default: 42)
            - enable_parallel: Enable parallel processing (default: True)
            - n_jobs: Number of parallel jobs (default: -1 for all cores)
            - save_models: Save trained models (default: False)
            - save_plots: Save visualization plots (default: False)
            - output_dir: Output directory (required if save_models or save_plots)
            - sample_lists_path: Path to sample lists (for ROSMAP-style splits)
            - use_predefined_splits: Use predefined train/val/test splits (default: False)

    Returns:
        Tuple of:
            - results_df: DataFrame with columns [region_id, region_name, chr, start, end,
                         n_variants, train_r2, train_pearson, val_r2, val_pearson,
                         test_r2, test_pearson, optimal_alpha, n_active_variants]
            - trained_models: Dict mapping region_id -> trained sklearn model

    Example:
        >>> config = {
        ...     'data_type': 'peaks',
        ...     'data_dir': './data/LCL',
        ...     'vcf_file_path': './data/chr22.vcf.gz',
        ...     'context_window': 100000,
        ...     'min_maf': 0.05,
        ...     'n_regions': 100,
        ...     'test_size': 0.2,
        ...     'enable_parallel': True
        ... }
        >>> results_df, models = run_predixcan_workflow(config)
    """
    # Validate configuration
    _validate_predixcan_config(config)

    # Step 1: Load and prepare data
    logger.info("Step 1/4: Loading data...")
    metadata, expr_data, sample_names, sample_lists = _prepare_predixcan_data(config)

    logger.info(f"Loaded {len(metadata)} regions with {len(sample_names)} samples")

    # Step 2: Prepare train/test splits
    logger.info("Step 2/4: Preparing train/test splits...")
    if config.get('use_predefined_splits', False) and sample_lists:
        train_samples = sample_lists['train_subs']
        val_samples = sample_lists.get('val_subs', [])
        test_samples = sample_lists['test_subs']
    else:
        train_samples, val_samples, test_samples = _create_splits(
            sample_names=sample_names,
            test_size=config.get('test_size', 0.2),
            val_size=config.get('val_size', 0.0),
            random_seed=config.get('random_seed', 42)
        )
        
        # Save split to file for reproducibility and use in inference
        if 'output_dir' in config:
            _save_train_test_split(
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                output_dir=config['output_dir'],
                test_size=config.get('test_size', 0.2),
                val_size=config.get('val_size', 0.0),
                random_seed=config.get('random_seed', 42)
            )

    logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

    # Step 3: Process regions (extract variants + train models)
    logger.info("Step 3/4: Training models...")
    results_list, trained_models = _train_models(
        metadata=metadata,
        expr_data=expr_data,
        sample_names=sample_names,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        config=config
    )

    # Step 4: Compile results and generate visualizations
    logger.info("Step 4/4: Compiling results...")
    results_df = pd.DataFrame(results_list)

    # Generate summary statistics
    summary = summarize_predixcan_results(results_list)
    logger.info(f"PrediXcan Summary:\n{summary}")

    # Save outputs if requested
    if config.get('save_models', False):
        _save_models(trained_models, config['output_dir'])
        logger.info(f"Saved models to {config['output_dir']}/models/")

    if config.get('save_plots', False):
        _create_and_save_plots(results_df, config['output_dir'])
        logger.info(f"Saved plots to {config['output_dir']}/figures/")

    logger.info(f"PrediXcan complete. Processed {len(results_df)} regions.")
    logger.info(f"Mean test R²: {results_df['test_r2'].mean():.3f}")
    logger.info(f"Mean test Pearson: {results_df['test_pearson'].mean():.3f}")

    return results_df, trained_models


def _validate_predixcan_config(config: Dict) -> None:
    """Validate required configuration parameters."""
    required = ['data_type', 'vcf_file_path']

    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config['data_type'] not in ['peaks', 'genes']:
        raise ValueError("data_type must be 'peaks' or 'genes'")

    if config['data_type'] == 'peaks' and 'data_dir' not in config:
        raise ValueError("data_dir required for data_type='peaks'")

    if config['data_type'] == 'genes':
        if 'gene_meta_path' not in config or 'expr_data_path' not in config:
            raise ValueError("gene_meta_path and expr_data_path required for data_type='genes'")

    if (config.get('save_models', False) or config.get('save_plots', False)) and 'output_dir' not in config:
        raise ValueError("output_dir required when save_models or save_plots is True")


def _prepare_predixcan_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[Dict]]:
    """
    Load and prepare data for PrediXcan.

    Returns:
        Tuple of (metadata, expr_data, sample_names, sample_lists)
    """
    if config['data_type'] == 'peaks':
        metadata, expr_data, sample_names = load_peaks(
            data_dir=config['data_dir'],
            n_peaks=config.get('n_regions'),
            selection_method=config.get('selection_method', 'variance'),
            random_state=config.get('random_seed', 42),
            start_rank=config.get('region_start_rank'),
            end_rank=config.get('region_end_rank')
        )
        sample_lists = None

    else:  # genes
        metadata, expr_data, sample_lists = load_genes(
            gene_meta_path=config['gene_meta_path'],
            expr_data_path=config['expr_data_path'],
            sample_lists_path=config.get('sample_lists_path'),
            select_genes=config.get('select_genes'),
            n_genes=config.get('n_regions'),
            selection_method=config.get('selection_method', 'variance'),
            random_state=config.get('random_seed', 42),
            start_rank=config.get('region_start_rank'),
            end_rank=config.get('region_end_rank')
        )
        sample_names = list(expr_data.columns)

    return metadata, expr_data, sample_names, sample_lists


def _create_splits(
    sample_names: List[str],
    test_size: float,
    val_size: float,
    random_seed: int
) -> Tuple[List[str], List[str], List[str]]:
    """Create train/val/test splits from sample names."""
    from sklearn.model_selection import train_test_split

    samples = np.array(sample_names)

    if val_size > 0:
        # Three-way split
        train_val, test = train_test_split(
            samples, test_size=test_size, random_state=random_seed
        )
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=random_seed
        )
        return train.tolist(), val.tolist(), test.tolist()
    else:
        # Two-way split
        train, test = train_test_split(
            samples, test_size=test_size, random_state=random_seed
        )
        return train.tolist(), [], test.tolist()


def _save_train_test_split(
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    output_dir: Union[str, Path],
    test_size: float,
    val_size: float,
    random_seed: int
) -> None:
    """
    Save train/val/test split to JSON file.
    
    This allows the same split to be reused across different analyses
    (e.g., PrediXcan and AlphaGenome inference on the same test set).
    """
    import json
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_file = output_dir / 'train_test_split.json'
    
    split_data = {
        'train_samples': train_samples,
        'test_samples': test_samples,
        'n_train': len(train_samples),
        'n_test': len(test_samples),
        'created_at': datetime.now().isoformat(),
        'metadata': {
            'test_size': test_size,
            'val_size': val_size,
            'random_seed': random_seed,
            'n_total_samples': len(train_samples) + len(val_samples) + len(test_samples)
        }
    }
    
    # Include validation samples if present
    if val_samples:
        split_data['val_samples'] = val_samples
        split_data['n_val'] = len(val_samples)
    
    with open(split_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Saved train/test split to: {split_file}")


def _train_models(
    metadata: pd.DataFrame,
    expr_data: pd.DataFrame,
    sample_names: List[str],
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    config: Dict
) -> Tuple[List[Dict], Dict]:
    """
    Train ElasticNet models for all regions.

    Returns:
        Tuple of (results_list, trained_models_dict)
    """
    enable_parallel = config.get('enable_parallel', True)
    n_jobs = config.get('n_jobs', -1)

    if enable_parallel and len(metadata) > 1 and n_jobs != 1:
        # Parallel processing with joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_region_predixcan)(
                region_data=(idx, row),
                expr_data=expr_data,
                sample_names=sample_names,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                config=config
            )
            for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Training models")
        )
    else:
        # Sequential processing
        results = []
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Training models"):
            result = _process_single_region_predixcan(
                region_data=(idx, row),
                expr_data=expr_data,
                sample_names=sample_names,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                config=config
            )
            results.append(result)

    # Separate results and models
    results_list = [r[0] for r in results if r[0] is not None]
    trained_models = {r[0]['region_id']: r[1] for r in results if r[0] is not None and r[1] is not None}

    return results_list, trained_models


def _process_single_region_predixcan(
    region_data: Tuple,
    expr_data: pd.DataFrame,
    sample_names: List[str],
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    config: Dict
) -> Tuple[Optional[Dict], Optional[object]]:
    """
    Process a single region: extract variants and train model.

    Returns:
        Tuple of (result_dict, trained_model)
    """
    idx, region = region_data
    region_id = region.get('ensg', idx)

    try:
        # Extract variants
        # Calculate region boundaries from TSS and context window
        context_window = config.get('context_window', 100000)
        half_window = context_window // 2
        start = max(0, region['tss'] - half_window)  # Ensure start is not negative
        end = region['tss'] + half_window

        genotype_matrix, variant_positions, variant_info = extract_variants_in_region(
            vcf_path=config['vcf_file_path'],
            chrom=region['chr'],
            start=start,
            end=end,
            samples=sample_names,
            min_maf=0.0,  # Apply MAF filtering separately below
            contig_prefix=config.get('contig_prefix', '')
        )

        if genotype_matrix.shape[1] == 0:
            logger.debug(f"No variants found for region {region_id}")
            return None, None

        # Filter by MAF
        min_maf = config.get('min_maf', 0.05)
        if min_maf > 0:
            genotype_matrix, kept_indices = filter_variants_by_maf(
                genotype_matrix=genotype_matrix,
                min_maf=min_maf
            )
            # Filter variant info and positions using kept indices
            variant_info = [variant_info[i] for i in kept_indices]
            variant_positions = [variant_positions[i] for i in kept_indices]

        if genotype_matrix.shape[1] == 0:
            logger.debug(f"No variants after MAF filtering for region {region_id}")
            return None, None

        # Get expression values
        y = expr_data.loc[region_id, sample_names].values

        # Prepare data splits
        train_idx = [i for i, s in enumerate(sample_names) if s in train_samples]
        test_idx = [i for i, s in enumerate(sample_names) if s in test_samples]
        val_idx = [i for i, s in enumerate(sample_names) if s in val_samples] if val_samples else None

        X_train, y_train = genotype_matrix[train_idx, :], y[train_idx]
        X_test, y_test = genotype_matrix[test_idx, :], y[test_idx]

        if val_idx:
            X_val, y_val = genotype_matrix[val_idx, :], y[val_idx]
        else:
            X_val, y_val = None, None

        # Train ElasticNet
        result = train_elasticnet(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            l1_ratio=config.get('l1_ratio', 0.5),
            alphas=config.get('alphas'),
            random_state=config.get('random_seed', 42)
        )

        # Extract model and metrics from result dictionary
        model = result['model']
        metrics = result

        # Count variants in/outside region
        region_start = region.get('Pos_Left', region.get('start', region['tss'] - 20000))
        region_end = region.get('Pos_Right', region.get('end', region['tss'] + 20000))

        variants_in_region = sum(1 for v in variant_info if region_start <= v.pos <= region_end)
        variants_outside_region = len(variant_info) - variants_in_region

        # Compile result
        result = {
            'region_id': region_id,
            'region_name': region.get('gene_name', str(region_id)),
            'chr': region['chr'],
            'start': region_start,
            'end': region_end,
            'tss': region['tss'],
            'n_variants': len(variant_info),
            'variants_in_region': variants_in_region,
            'variants_outside_region': variants_outside_region,
            'train_r2': metrics['train_r2'],
            'train_pearson': metrics['train_pearson'],
            'val_r2': metrics.get('val_r2', np.nan),
            'val_pearson': metrics.get('val_pearson', np.nan),
            'test_r2': metrics['test_r2'],
            'test_pearson': metrics['test_pearson'],
            'optimal_alpha': metrics.get('optimal_alpha', np.nan),
            'n_active_variants': metrics.get('n_active_variants', 0)
        }

        return result, model if config.get('save_models', False) else None

    except Exception as e:
        logger.error(f"Failed to process region {region_id}: {str(e)}")
        return None, None


def _save_models(models: Dict, output_dir: str) -> None:
    """Save trained models to disk."""
    import joblib

    output_path = Path(output_dir) / "models"
    output_path.mkdir(parents=True, exist_ok=True)

    for region_id, model in models.items():
        save_path = output_path / f"{region_id}_model.joblib"
        joblib.dump(model, save_path)


def _create_and_save_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create and save comprehensive visualization plots."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    # Use the comprehensive plotting function from utils
    create_comprehensive_predixcan_plots(
        results_df=results_df,
        output_dir=str(output_path)
    )
