"""
ATAC-Based PrediXcan Workflow

This module provides a workflow for training ElasticNet models to predict
gene expression from binned predicted ATAC-seq signals.

The approach:
1. For each gene, predict ATAC signal around TSS for all samples
2. Bin predictions to fixed-size bins (256bp default, 390 bins for 100KB)
3. Apply log1p + StandardScaler normalization
4. Train ElasticNet to predict expression from binned ATAC features
5. Evaluate on train/val/test splits

This is an alternative to variant-based PrediXcan that uses chromatin
accessibility predictions as features instead of genotype data.

Author: AlphaGenome Evaluation Team
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from alphagenome_eval.utils import (
    load_genes,
    create_genome_dataset,
    train_elasticnet,
    evaluate_predictions,
    summarize_predixcan_results,
    create_comprehensive_predixcan_plots,
)

from alphagenome_eval.utils.binning import (
    create_feature_matrix_for_gene,
    DEFAULT_BIN_SIZE,
    DEFAULT_WINDOW_SIZE,
)

logger = logging.getLogger(__name__)


def run_atac_predixcan_workflow(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete ATAC-based PrediXcan workflow.

    This function orchestrates the pipeline:
    1. Load gene metadata and expression data
    2. Initialize ATAC prediction model (AlphaGenome or Borzoi)
    3. Create GenomeDataset for efficient sequence extraction
    4. Extract binned ATAC features for all genes x samples
    5. Train ElasticNet models for each gene
    6. Evaluate and save results

    Args:
        config: Configuration dictionary with keys:
            # Data paths
            - gene_meta_path: Path to gene metadata
            - expr_data_path: Path to expression data
            - sample_lists_path: Path to sample lists
            - vcf_file_path: Path to VCF file
            - hg38_file_path: Path to reference genome

            # Model configuration
            - model_type: 'alphagenome' or 'borzoi' (default: 'borzoi')
            - api_key: AlphaGenome API key (if using alphagenome)
            - ontology_terms: Tissue ontology terms (AlphaGenome)
            - borzoi_tissue_query: Tissue query for Borzoi DNASE tracks
            - use_flashzoi: Use Flashzoi for faster inference (default: True)
            - borzoi_replicate: Model replicate (default: 0)

            # Feature extraction
            - window_size: Window around TSS in bp (default: 100000)
            - bin_size: Target bin size in bp (default: 256)

            # Training
            - n_genes: Number of genes to analyze (optional)
            - selection_method: Gene selection method (default: 'variance')
            - use_predefined_splits: Use predefined train/val/test (default: True)
            - l1_ratio: ElasticNet L1 ratio (default: 0.5)
            - random_seed: Random seed (default: 42)

            # Output
            - output_dir: Output directory
            - save_models: Save trained models (default: True)
            - save_plots: Save visualization plots (default: True)
            - cache_predictions: Cache ATAC predictions (default: True)
            - save_all_predictions: Save all binned predictions to NPZ (default: False)
            - predictions_output_path: Custom path for predictions file (optional)

    Returns:
        Tuple of:
            - results_df: DataFrame with per-gene metrics
            - trained_models: Dict of gene_id -> trained model

    Example:
        >>> config = {
        ...     'gene_meta_path': 'gene-ids-and-positions.tsv',
        ...     'expr_data_path': 'expression.csv',
        ...     'vcf_file_path': 'genotypes.vcf.gz',
        ...     'hg38_file_path': 'hg38.fa',
        ...     'model_type': 'borzoi',
        ...     'borzoi_tissue_query': 'brain',
        ...     'window_size': 100000,
        ...     'bin_size': 256,
        ...     'n_genes': 50,
        ...     'output_dir': './results/atac_predixcan'
        ... }
        >>> results_df, models = run_atac_predixcan_workflow(config)
    """
    # Validate configuration
    _validate_atac_predixcan_config(config)

    # Create output directory
    output_dir = Path(config.get('output_dir', './results/atac_predixcan'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    logger.info("Step 1/5: Loading gene data and expression...")
    metadata, expr_data, sample_lists = _prepare_atac_predixcan_data(config)
    sample_names = list(expr_data.columns)

    logger.info(f"Loaded {len(metadata)} genes with {len(sample_names)} samples")

    # Step 2: Prepare train/val/test splits
    logger.info("Step 2/5: Preparing train/val/test splits...")
    if config.get('use_predefined_splits', True) and sample_lists:
        train_samples = sample_lists['train_subs']
        val_samples = sample_lists.get('val_subs', [])
        test_samples = sample_lists['test_subs']
    else:
        train_samples, val_samples, test_samples = _create_splits(
            sample_names=sample_names,
            test_size=config.get('test_size', 0.2),
            val_size=config.get('val_size', 0.1),
            random_seed=config.get('random_seed', 42)
        )

    logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

    # Save split info
    _save_split_info(
        train_samples, val_samples, test_samples,
        output_dir, config
    )

    # Step 3: Initialize ATAC model
    logger.info("Step 3/5: Initializing ATAC prediction model...")
    model, track_indices, device = _initialize_atac_model(config)

    # Step 4: Create genome dataset
    logger.info("Step 4/5: Creating genome dataset...")
    genome_dataset = _create_genome_dataset(config, metadata, expr_data, sample_names)

    # Prepare config for feature extraction
    feature_config = {
        'window_size': config.get('window_size', DEFAULT_WINDOW_SIZE),
        'target_bin_size': config.get('bin_size', DEFAULT_BIN_SIZE),
        'aggregation': config.get('aggregation', 'mean'),
        'track_indices': track_indices,
        'ontology_terms': config.get('ontology_terms', []),
        'sequence_length_bp': config.get('sequence_length_bp', 102400),
        'alphagenome_output_type': config.get('alphagenome_output_type', 'dnase'),  # DNASE has more brain tracks
    }

    # Step 5: Process each gene
    logger.info("Step 5/5: Extracting features and training models...")
    results_list = []
    trained_models = {}

    # Determine samples to process (combine all splits)
    all_samples = train_samples + val_samples + test_samples

    # Check if per-track models are requested (Borzoi only, default: True)
    per_track_models = config.get('per_track_models', True)
    use_per_track_mode = per_track_models and config.get('model_type', 'borzoi') == 'borzoi' and track_indices

    if use_per_track_mode:
        # Get track descriptions for logging
        from alphagenome_eval.utils import init_borzoi_model
        _, tracks_df = init_borzoi_model(
            device=device,
            use_flashzoi=config.get('use_flashzoi', True),
            replicate=config.get('borzoi_replicate', 0)
        )
        track_descs = {idx: tracks_df.loc[idx, 'description'] for idx in track_indices}
        logger.info(f"Per-track mode: Training separate models for {len(track_indices)} tracks")
        logger.info(f"Efficient mode: Running inference ONCE per gene, then splitting per track")

    cache_dir = output_dir / 'cache' if config.get('cache_predictions', True) else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize collectors for saving all predictions
    all_predictions_collector = {} if config.get('save_all_predictions', False) else None
    all_valid_samples_collector = {} if config.get('save_all_predictions', False) else None

    # Determine total iterations for progress bar
    n_tracks = len(track_indices) if use_per_track_mode else 1
    total_iterations = len(metadata) * n_tracks
    pbar = tqdm(total=total_iterations, desc="Processing genes")

    # Generate cache key for feature extraction
    cache_config_str = (
        f"{feature_config.get('window_size', DEFAULT_WINDOW_SIZE)}_"
        f"{feature_config.get('target_bin_size', DEFAULT_BIN_SIZE)}_"
        f"{feature_config.get('aggregation', 'mean')}_"
        f"{config.get('model_type', 'borzoi')}"
    )
    if use_per_track_mode:
        cache_config_str += f"_pertrack_{'-'.join(map(str, track_indices))}"
    cache_key = hashlib.md5(cache_config_str.encode()).hexdigest()[:8]
    logger.info(f"Feature cache key: {cache_key}")

    # Process each gene (outer loop)
    for idx, row in metadata.iterrows():
        gene_id = row.get('ensg', row.get('gene_id', idx))

        try:
            # Extract features ONCE for all tracks
            if cache_dir:
                cache_file = cache_dir / f"{gene_id}_{cache_key}_features.npz"
                if cache_file.exists():
                    cached = np.load(cache_file, allow_pickle=True)
                    X_all = cached['X']
                    valid_samples = cached['sample_ids'].tolist()
                    logger.debug(f"Loaded cached features for {gene_id}")

                    # Store for combined output (from cache)
                    if all_predictions_collector is not None:
                        all_predictions_collector[gene_id] = X_all.copy()
                        all_valid_samples_collector[gene_id] = valid_samples.copy()
                else:
                    # Extract features with return_per_track=True for efficient per-track mode
                    X_all, valid_samples = create_feature_matrix_for_gene(
                        model=model,
                        genome_dataset=genome_dataset,
                        gene_row=row.to_dict(),
                        sample_ids=all_samples,
                        model_type=config.get('model_type', 'borzoi'),
                        config=feature_config,
                        device=device,
                        return_per_track=use_per_track_mode
                    )
                    # Cache features
                    np.savez(cache_file, X=X_all, sample_ids=np.array(valid_samples))

                # Store for combined output
                if all_predictions_collector is not None:
                    all_predictions_collector[gene_id] = X_all.copy()
                    all_valid_samples_collector[gene_id] = valid_samples.copy()
            else:
                # Extract features without caching
                X_all, valid_samples = create_feature_matrix_for_gene(
                    model=model,
                    genome_dataset=genome_dataset,
                    gene_row=row.to_dict(),
                    sample_ids=all_samples,
                    model_type=config.get('model_type', 'borzoi'),
                    config=feature_config,
                    device=device,
                    return_per_track=use_per_track_mode
                )

                # Store for combined output (no caching case)
                if all_predictions_collector is not None:
                    all_predictions_collector[gene_id] = X_all.copy()
                    all_valid_samples_collector[gene_id] = valid_samples.copy()

            if len(valid_samples) < 10:
                logger.warning(f"Not enough valid samples for {gene_id}: {len(valid_samples)}")
                pbar.update(n_tracks)
                continue

            # Get expression values for valid samples
            y = expr_data.loc[gene_id, valid_samples].values

            # Create sample index mapping
            sample_to_idx = {s: i for i, s in enumerate(valid_samples)}

            # Filter to valid samples in each split
            train_idx = [sample_to_idx[s] for s in train_samples if s in sample_to_idx]
            val_idx = [sample_to_idx[s] for s in val_samples if s in sample_to_idx]
            test_idx = [sample_to_idx[s] for s in test_samples if s in sample_to_idx]

            if len(train_idx) < 5 or len(test_idx) < 3:
                logger.warning(f"Not enough samples in splits for {gene_id}")
                pbar.update(n_tracks)
                continue

            # Determine tracks to process
            if use_per_track_mode:
                # X_all shape: (n_samples, n_tracks, n_bins)
                tracks_to_train = [(i, track_indices[i], track_descs[track_indices[i]])
                                   for i in range(len(track_indices))]
            else:
                # X_all shape: (n_samples, n_bins)
                tracks_to_train = [(None, None, 'all_tracks_averaged')]

            # Train model for each track (inner loop - no additional inference!)
            for track_local_idx, track_idx, track_desc in tracks_to_train:
                pbar.update(1)

                try:
                    # Extract features for this track
                    if track_local_idx is not None:
                        # Per-track mode: extract from 3D array
                        X = X_all[:, track_local_idx, :]  # (n_samples, n_bins)
                    else:
                        # Averaged mode: use 2D array directly
                        X = X_all  # (n_samples, n_bins)

                    # Apply normalization: log1p + StandardScaler
                    X_log = np.log1p(X)

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_log[train_idx])
                    X_test = scaler.transform(X_log[test_idx])

                    if val_idx:
                        X_val = scaler.transform(X_log[val_idx])
                        y_val = y[val_idx]
                    else:
                        X_val, y_val = None, None

                    y_train = y[train_idx]
                    y_test = y[test_idx]

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

                    # Compile result with track info
                    gene_result = {
                        'gene_id': gene_id,
                        'gene_name': row.get('gene_name', str(gene_id)),
                        'chr': row['chr'],
                        'tss': row['tss'],
                        'track_idx': track_idx if track_idx is not None else 'averaged',
                        'track_desc': track_desc,
                        'n_samples_train': len(train_idx),
                        'n_samples_val': len(val_idx),
                        'n_samples_test': len(test_idx),
                        'n_bins': X.shape[1],
                        'train_r2': result['train_r2'],
                        'train_pearson': result['train_pearson'],
                        'val_r2': result.get('val_r2', np.nan),
                        'val_pearson': result.get('val_pearson', np.nan),
                        'test_r2': result['test_r2'],
                        'test_pearson': result['test_pearson'],
                        'optimal_alpha': result.get('optimal_alpha', np.nan),
                        'n_active_features': result.get('n_active_variants', 0)
                    }

                    results_list.append(gene_result)

                    # Store model with track info if per-track
                    model_key = f"{gene_id}_track{track_idx}" if track_idx is not None else gene_id
                    if config.get('save_models', True):
                        trained_models[model_key] = {
                            'model': result['model'],
                            'scaler': scaler,
                            'track_idx': track_idx,
                            'track_desc': track_desc
                        }

                except Exception as e:
                    logger.error(f"Failed to train model for gene {gene_id} (track {track_idx}): {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Failed to process gene {gene_id}: {str(e)}")
            pbar.update(n_tracks)
            continue

    pbar.close()

    # Compile results
    results_df = pd.DataFrame(results_list)

    if len(results_df) == 0:
        logger.warning("No genes processed successfully")
        return results_df, trained_models

    # Save results
    results_file = output_dir / 'atac_predixcan_results.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved results to {results_file}")

    # Save metadata
    _save_metadata(config, results_df, output_dir)

    # Save all predictions if requested
    if all_predictions_collector:
        from alphagenome_eval.utils.binning import save_all_predictions

        # Build track info
        if use_per_track_mode:
            track_info_dict = {
                'indices': track_indices,
                'descriptions': [track_descs.get(idx, '') for idx in track_indices]
            }
        else:
            track_info_dict = {
                'ontology_terms': config.get('ontology_terms', [])
            }

        predictions_path = save_all_predictions(
            output_path=config.get('predictions_output_path',
                                    output_dir / 'all_predictions.npz'),
            predictions_dict=all_predictions_collector,
            gene_ids=list(all_predictions_collector.keys()),
            sample_ids=all_samples,
            track_info=track_info_dict,
            config=feature_config,
            gene_metadata=metadata,
            model_type=config.get('model_type', 'borzoi'),
            valid_samples_per_gene=all_valid_samples_collector
        )
        logger.info(f"Saved all predictions to {predictions_path}")

    # Save models if requested
    if config.get('save_models', True) and trained_models:
        _save_models(trained_models, output_dir)
        logger.info(f"Saved {len(trained_models)} models to {output_dir}/models/")

    # Generate plots if requested
    if config.get('save_plots', True):
        _create_and_save_plots(results_df, output_dir)
        logger.info(f"Saved plots to {output_dir}/figures/")

    # Print summary
    logger.info(f"ATAC-PrediXcan complete. Processed {len(results_df)} genes.")
    logger.info(f"Mean test R²: {results_df['test_r2'].mean():.3f}")
    logger.info(f"Mean test Pearson: {results_df['test_pearson'].mean():.3f}")

    return results_df, trained_models


def _validate_atac_predixcan_config(config: Dict) -> None:
    """Validate required configuration parameters."""
    required = ['gene_meta_path', 'expr_data_path', 'vcf_file_path', 'hg38_file_path']

    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    model_type = config.get('model_type', 'borzoi')
    if model_type not in ['borzoi', 'alphagenome']:
        raise ValueError("model_type must be 'borzoi' or 'alphagenome'")

    if model_type == 'alphagenome' and 'api_key' not in config:
        # Check environment variable
        import os
        if 'ALPHAGENOME_API_KEY' not in os.environ:
            raise ValueError("api_key required for model_type='alphagenome'")

    if config.get('save_models', True) or config.get('save_plots', True):
        if 'output_dir' not in config:
            raise ValueError("output_dir required when save_models or save_plots is True")


def _prepare_atac_predixcan_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict]]:
    """Load and prepare data for ATAC-based PrediXcan."""
    metadata, expr_data, sample_lists = load_genes(
        gene_meta_path=config['gene_meta_path'],
        expr_data_path=config['expr_data_path'],
        sample_lists_path=config.get('sample_lists_path'),
        select_genes=config.get('select_genes'),
        n_genes=config.get('n_genes'),
        selection_method=config.get('selection_method', 'variance'),
        random_state=config.get('random_seed', 42),
        start_rank=config.get('region_start_rank'),
        end_rank=config.get('region_end_rank'),
        predixcan_results_path=config.get('predixcan_results_path'),
        predixcan_metric=config.get('predixcan_metric', 'val_pearson')
    )

    return metadata, expr_data, sample_lists


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


def _save_split_info(
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    output_dir: Path,
    config: Dict
) -> None:
    """Save train/val/test split information."""
    split_file = output_dir / 'train_test_split.json'

    split_data = {
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_test': len(test_samples),
        'created_at': datetime.now().isoformat(),
        'use_predefined_splits': config.get('use_predefined_splits', True)
    }

    with open(split_file, 'w') as f:
        json.dump(split_data, f, indent=2)

    logger.info(f"Saved split info to {split_file}")


def _initialize_atac_model(config: Dict) -> Tuple[object, List[int], str]:
    """Initialize ATAC prediction model (Borzoi or AlphaGenome)."""
    model_type = config.get('model_type', 'borzoi')

    if model_type == 'borzoi':
        from alphagenome_eval.utils import (
            init_borzoi_model,
            get_borzoi_dnase_track_indices,
            select_device
        )

        device = select_device()
        use_flashzoi = config.get('use_flashzoi', True)
        replicate = config.get('borzoi_replicate', 0)
        use_random_init = config.get('use_random_init', False)

        model, tracks_df = init_borzoi_model(
            device=device,
            use_flashzoi=use_flashzoi,
            replicate=replicate,
            use_random_init=use_random_init
        )

        # Get DNASE track indices - either custom or by tissue query
        custom_track_indices = config.get('borzoi_track_indices', None)

        if custom_track_indices is not None:
            # Use custom track indices provided in config
            track_indices = list(custom_track_indices)
            # Log track descriptions for verification
            track_descs = tracks_df.loc[track_indices, 'description'].tolist()
            logger.info(f"Using {len(track_indices)} custom track indices: {track_indices}")
            for idx, desc in zip(track_indices, track_descs):
                logger.info(f"  [{idx}] {desc}")
        else:
            # Use tissue query to find tracks
            tissue_query = config.get('borzoi_tissue_query', 'brain')
            track_indices = get_borzoi_dnase_track_indices(
                tracks_df=tracks_df,
                tissue_query=tissue_query
            )

            if not track_indices:
                logger.warning(f"No DNASE tracks found for '{tissue_query}', using all DNASE tracks")
                track_indices = get_borzoi_dnase_track_indices(tracks_df=tracks_df)

            logger.info(f"Initialized Borzoi with {len(track_indices)} DNASE tracks for '{tissue_query}'")

        return model, track_indices, device

    else:  # alphagenome
        from alphagenome_eval.utils import init_dna_model
        import os

        api_key = config.get('api_key', os.environ.get('ALPHAGENOME_API_KEY'))
        sequence_length = config.get('sequence_length', '100KB')

        model, seq_len = init_dna_model(
            api_key=api_key,
            sequence_length=sequence_length
        )

        config['sequence_length_bp'] = seq_len

        logger.info(f"Initialized AlphaGenome with sequence length {seq_len}")

        return model, [], 'cpu'


def _create_genome_dataset(
    config: Dict,
    metadata: pd.DataFrame,
    expr_data: pd.DataFrame,
    sample_names: List[str]
) -> object:
    """Create GenomeDataset for efficient sequence extraction."""
    from alphagenome_eval import GenomeDataset

    # Determine sequence length based on model type
    model_type = config.get('model_type', 'borzoi')

    if model_type == 'borzoi':
        from alphagenome_eval.utils import BORZOI_SEQ_LEN
        window_size = BORZOI_SEQ_LEN  # 524288
    else:
        window_size = config.get('sequence_length_bp', 102400)

    # Create GenomeDataset directly with only_personal=True to get mat/pat sequences
    genome_dataset = GenomeDataset(
        gene_metadata=metadata,
        vcf_file_path=config['vcf_file_path'],
        hg38_file_path=config['hg38_file_path'],
        expr_data=expr_data,
        sample_list=sample_names,
        window_size=window_size,
        verbose=False,
        only_personal=True,  # Only return personal sequences (mat, pat)
        onehot_encode=False,  # Return string sequences
        contig_prefix='',  # VCF uses no prefix
        genome_contig_prefix='chr'  # hg38 uses 'chr' prefix
    )

    return genome_dataset


def _save_metadata(config: Dict, results_df: pd.DataFrame, output_dir: Path) -> None:
    """Save run metadata to JSON file."""
    metadata_file = output_dir / 'metadata.json'

    # Remove non-serializable items
    serializable_config = {}
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable_config[k] = v
        else:
            serializable_config[k] = str(v)

    metadata = {
        'config': serializable_config,
        'n_genes_processed': len(results_df),
        'mean_test_r2': float(results_df['test_r2'].mean()),
        'mean_test_pearson': float(results_df['test_pearson'].mean()),
        'created_at': datetime.now().isoformat()
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def _save_models(models: Dict, output_dir: Path) -> None:
    """Save trained models to disk."""
    import joblib

    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    for gene_id, model_data in models.items():
        save_path = models_dir / f"{gene_id}_model.joblib"
        joblib.dump(model_data, save_path)


def _create_and_save_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save visualization plots."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Rename columns to match expected format for plotting
        plot_df = results_df.rename(columns={
            'gene_id': 'region_id',
            'gene_name': 'region_name',
            'n_bins': 'n_variants'  # For compatibility with existing plotting
        })

        create_comprehensive_predixcan_plots(
            results_df=plot_df,
            output_dir=str(figures_dir)
        )
    except Exception as e:
        logger.warning(f"Failed to create some plots: {e}")


def run_atac_predixcan_multi_window(
    config: Dict,
    window_sizes: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """
    Run ATAC-based PrediXcan workflow with multiple window sizes.

    This wrapper function runs the workflow for each specified window size
    and generates a comparison summary.

    Args:
        config: Base configuration dictionary (see run_atac_predixcan_workflow)
        window_sizes: List of window sizes to test (default: [50000, 100000, 200000])
                     Each size represents the total window around TSS in bp.

    Returns:
        Tuple of:
            - combined_results: DataFrame with results from all window sizes,
                               includes 'window_size' column for comparison
            - all_models: Dict of {window_size: trained_models_dict}

    Example:
        >>> config = {
        ...     'gene_meta_path': 'gene-ids-and-positions.tsv',
        ...     'expr_data_path': 'expression.csv',
        ...     'vcf_file_path': 'genotypes.vcf.gz',
        ...     'hg38_file_path': 'hg38.fa',
        ...     'model_type': 'borzoi',
        ...     'n_genes': 50,
        ...     'output_dir': './results/atac_predixcan'
        ... }
        >>> results, models = run_atac_predixcan_multi_window(
        ...     config, window_sizes=[50000, 100000, 200000]
        ... )
        >>> # Compare performance across window sizes
        >>> results.groupby('window_size')['test_pearson'].mean()
    """
    if window_sizes is None:
        window_sizes = [50000, 100000, 200000]  # 50KB, 100KB, 200KB

    base_output_dir = Path(config.get('output_dir', './results/atac_predixcan'))
    all_results = []
    all_models = {}

    logger.info(f"Running ATAC PrediXcan with window sizes: {window_sizes}")

    for window_size in window_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing window size: {window_size:,} bp ({window_size//1000}KB)")
        logger.info(f"{'='*60}")

        # Create window-specific config
        window_config = config.copy()
        window_config['window_size'] = window_size
        window_config['output_dir'] = str(base_output_dir / f"window_{window_size//1000}kb")

        # Run workflow for this window size
        try:
            results_df, trained_models = run_atac_predixcan_workflow(window_config)

            if len(results_df) > 0:
                # Add window size column
                results_df = results_df.copy()
                results_df['window_size'] = window_size
                results_df['window_size_kb'] = window_size // 1000
                all_results.append(results_df)
                all_models[window_size] = trained_models

                logger.info(f"Window {window_size//1000}KB: "
                           f"Mean test Pearson = {results_df['test_pearson'].mean():.3f}")
            else:
                logger.warning(f"No results for window size {window_size}")

        except Exception as e:
            logger.error(f"Failed to process window size {window_size}: {e}")
            continue

    if not all_results:
        logger.error("No results from any window size")
        return pd.DataFrame(), {}

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save combined results
    combined_file = base_output_dir / 'combined_multi_window_results.csv'
    combined_results.to_csv(combined_file, index=False)
    logger.info(f"Saved combined results to {combined_file}")

    # Generate comparison summary
    summary = combined_results.groupby('window_size_kb').agg({
        'test_pearson': ['mean', 'std', 'median'],
        'test_r2': ['mean', 'std', 'median'],
        'n_active_features': 'mean',
        'gene_id': 'count'
    }).round(4)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'gene_id_count': 'n_genes'})

    summary_file = base_output_dir / 'window_size_comparison.csv'
    summary.to_csv(summary_file)
    logger.info(f"Saved window size comparison to {summary_file}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Window Size Comparison Summary")
    logger.info("="*60)
    logger.info(f"\n{summary.to_string()}")

    return combined_results, all_models


__all__ = [
    'run_atac_predixcan_workflow',
    'run_atac_predixcan_multi_window',
]
