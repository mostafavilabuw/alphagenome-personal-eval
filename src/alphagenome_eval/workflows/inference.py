"""
Inference Workflow for AlphaGenome Predictions

This module provides a unified workflow for running AlphaGenome predictions
across multiple genomic regions (peaks or genes) and calculating correlations
with observed data.

Consolidates:
- explore/LCL/inference/inference_multi_tissue.py (multi-tissue ATAC)
- explore/ROSMAP/inference/inference_rosmap.py (gene expression)

Author: AlphaGenome Evaluation Team
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

if TYPE_CHECKING:
    from alphagenome_eval.PersonalDataset import GenomeDataset

from alphagenome_eval.utils import (
    load_peaks,
    load_genes,
    create_genome_dataset,
    init_dna_model,
    predict_personal_genome_atac,
    predict_personal_genome_rna,
    batch_predict_regions,
    calculate_prediction_statistics,
    get_personal_sequences,
    # Borzoi utilities
    init_borzoi_model,
    predict_borzoi,
    get_borzoi_track_indices,
    predict_borzoi_personal_genome,
    aggregate_borzoi_over_region,
    BORZOI_SEQ_LEN,
    BORZOI_PRED_LENGTH,
    BORZOI_SUPPORTED_CONTEXT_WINDOWS,
    validate_borzoi_context_window,
    # Borzoi RNA paired strand prediction (for ROSMAP brain)
    get_borzoi_rna_track_by_identifier,
    predict_borzoi_personal_rna_paired_strand,
    # Borzoi multi-track CSV support
    load_tracks_from_csv,
    predict_borzoi_multi_track,
    # GPU utilities
    select_device,
)
from alphagenome_eval.utils.visualization import create_inference_plots

logger = logging.getLogger(__name__)


def run_inference_workflow(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete inference workflow for AlphaGenome predictions.

    This function orchestrates the entire inference pipeline:
    1. Load genomic regions (peaks/genes) and expression data
    2. Initialize AlphaGenome model
    3. Run predictions across regions and samples
    4. Calculate correlations between predictions and observations
    5. Return results as DataFrames

    Args:
        config: Configuration dictionary with keys:
            - data_type: 'peaks' or 'genes'
            - api_key: AlphaGenome API key
            - data_dir: Path to data directory (for peaks)
            - gene_meta_path: Path to gene metadata (for genes)
            - expr_data_path: Path to expression data (for genes)
            - vcf_file_path: Path to VCF file
            - hg38_file_path: Path to reference genome
            - ontology_terms: List of tissue ontology terms
            - sequence_length: AlphaGenome sequence length (default: 1MB)
            - n_regions: Number of regions to process (optional)
            - n_samples: Number of samples per region (optional)
            - output_type: 'atac' or 'rna' (default: from data_type)
            - enable_parallel: Enable parallel processing (default: True)
            - max_workers: Max parallel workers (default: 8)
            - save_predictions: Save raw predictions (default: False)
            - output_dir: Output directory for predictions (required if save_predictions=True)
            - output_dir: Output directory for predictions (required if save_predictions=True)
            - random_seed: Random seed (default: 42)
            - model_type: 'alphagenome' or 'borzoi' (default: 'alphagenome')
            - borzoi_model_name: HuggingFace model name for Borzoi (optional)
            - use_flashzoi: Use Flashzoi (FlashAttention-2) for ~1.6x faster inference (default: False)
            - borzoi_replicate: Borzoi/Flashzoi replicate number 0-3 (default: 0)

    Returns:
        Tuple of:
            - results_df: DataFrame with columns [region_id, region_name, chr, start,
                         end, n_samples, pearson_corr, p_value, mean_pred, std_pred]
            - predictions: Dict mapping region_id -> {
                'predictions': np.array (samples x 1),
                'observed': np.array (samples x 1),
                'sample_ids': list
              }

    Example:
        >>> config = {
        ...     'data_type': 'peaks',
        ...     'api_key': 'YOUR_API_KEY',
        ...     'data_dir': './data/LCL',
        ...     'vcf_file_path': './data/chr22.vcf.gz',
        ...     'hg38_file_path': './data/hg38.fa',
        ...     'ontology_terms': ['EFO:0002067'],  # LCL
        ...     'n_regions': 100,
        ...     'n_samples': 50
        ... }
        >>> results_df, predictions = run_inference_workflow(config)
    """
    # Validate configuration
    _validate_config(config)

    # Step 1: Load and prepare data
    logger.info("Step 1/5: Loading data...")
    metadata, expr_data, sample_names, sample_lists = _prepare_inference_data(config)

    logger.info(f"Loaded {len(metadata)} regions with {len(sample_names)} samples")

    # Step 2: Initialize Model
    model_type = config.get('model_type', 'alphagenome')
    logger.info(f"Step 2/5: Initializing {model_type} model...")
    
    tracks_df = None
    borzoi_device = None  # Track device used for Borzoi model
    if model_type == 'borzoi':
        # Support both explicit model_name or use_flashzoi/replicate params
        model_name = config.get('borzoi_model_name')
        use_flashzoi = config.get('use_flashzoi', False)
        replicate = config.get('borzoi_replicate', 0)
        
        # Determine device: use config if specified, otherwise auto-select
        # Note: Resolve 'auto' here (not in init_borzoi_model) so we can store
        # the concrete device string in config['_borzoi_device'] for downstream use
        if config.get('device'):
            borzoi_device = config['device']
            if borzoi_device == 'auto':
                borzoi_device = select_device(prefer_cuda=True)
        elif config.get('use_cuda', True):
            borzoi_device = select_device(prefer_cuda=True)
        else:
            borzoi_device = 'cpu'
        
        model, tracks_df = init_borzoi_model(
            model_name=model_name,
            device=borzoi_device,
            use_flashzoi=use_flashzoi,
            replicate=replicate
        )
        sequence_length_bp = BORZOI_SEQ_LEN  # Fixed 524KB for Borzoi model input
        # Store effective device in config for downstream prediction calls
        config['_borzoi_device'] = borzoi_device
        logger.info(f"Borzoi model loaded on device: {borzoi_device}")
        
        # Handle context window for Borzoi (controls how much genomic sequence to extract)
        # Sequences shorter than 524KB are automatically center-padded with 'N' in dna_to_onehot()
        borzoi_context_window = config.get('borzoi_context_window', BORZOI_SEQ_LEN)
        if borzoi_context_window != BORZOI_SEQ_LEN:
            borzoi_context_window = validate_borzoi_context_window(borzoi_context_window)
            logger.info(f"Using context window: {borzoi_context_window} bp ({borzoi_context_window // 1024}KB)")
            logger.info(f"  Sequences will be center-padded with 'N' to {BORZOI_SEQ_LEN} bp for model input")
        config['_borzoi_context_window'] = borzoi_context_window
        
        # Load multi-track CSV if configured
        borzoi_tracks_csv = config.get('borzoi_tracks_csv')
        if borzoi_tracks_csv:
            all_indices, track_info, csv_output_type = load_tracks_from_csv(borzoi_tracks_csv)
            config['_borzoi_multi_track'] = {
                'all_indices': all_indices,
                'track_info': track_info,
                'output_type': csv_output_type
            }
            logger.info(f"Loaded {len(track_info)} tracks from {borzoi_tracks_csv}")
            logger.info(f"  Track type: {csv_output_type}, indices: {len(all_indices)}")
    else:
        sequence_length_str = config.get('sequence_length')
        if sequence_length_str:
            model, sequence_length_bp = init_dna_model(config['api_key'], sequence_length=sequence_length_str)
        else:
            model, sequence_length_bp = init_dna_model(config['api_key'])
            
    logger.info(f"Model initialized with sequence length: {sequence_length_bp} bp")

    # Step 3: Create GenomeDataset for efficient sequence extraction (optional optimization)
    genome_dataset = None
    if config.get('use_genome_dataset', False):
        logger.info("Step 3/6: Creating shared GenomeDataset for optimized sequence extraction...")
        
        # Determine target samples
        if sample_lists and config.get('target_sample_set'):
            target_samples = sample_lists[config['target_sample_set']]
        else:
            target_samples = sample_names[:config.get('n_samples', len(sample_names))]
        
        # For Borzoi, use context window size for sequence extraction
        # (shorter sequences are center-padded with 'N' during one-hot encoding)
        dataset_window_size = config.get('_borzoi_context_window', sequence_length_bp)
        
        # Create shared dataset instance
        genome_dataset = create_genome_dataset(
            metadata=metadata,
            vcf_file_path=config['vcf_file_path'],
            genome_file_path=config['hg38_file_path'],
            expression_data=expr_data,
            sample_list=target_samples,
            window_size=dataset_window_size,
            verbose=False,
            contig_prefix=config.get('contig_prefix', ''),
            genome_contig_prefix='chr',
            onehot_encode=False,
            return_idx=False
        )
        logger.info(f"Created GenomeDataset: {len(metadata)} genes × {len(target_samples)} samples")
        logger.info(f"  Window size: {dataset_window_size} bp ({dataset_window_size // 1024}KB)")
        logger.info(f"  Expected speedup: ~{len(target_samples)}x for sequence extraction (lazy caching)")

    # Step 4: Run predictions
    step_num = "4/6" if genome_dataset is not None else "3/5"
    logger.info(f"Step {step_num}: Running predictions...")
    results_list, predictions = _run_predictions(
        metadata=metadata,
        expr_data=expr_data,
        sample_names=sample_names,
        sample_lists=sample_lists,
        model=model,
        sequence_length_bp=sequence_length_bp,
        config=config,
        tracks_df=tracks_df,
        genome_dataset=genome_dataset
    )

    # Step 5: Compile results
    step_num = "5/6" if genome_dataset is not None else "4/5"
    logger.info(f"Step {step_num}: Compiling results...")
    results_df = pd.DataFrame(results_list)

    # Save predictions if requested
    if config.get('save_predictions', False):
        _save_predictions(predictions, config['output_dir'])
        logger.info(f"Saved predictions to {config['output_dir']}")
    
    # Check if multi-tissue ATAC mode and save per-tissue results
    ontology_terms = config.get('ontology_terms', [])
    output_type = config.get('output_type', 'atac' if config['data_type'] == 'peaks' else 'rna')
    is_multi_tissue_atac = (output_type == 'atac' and len(ontology_terms) > 1)
    
    if is_multi_tissue_atac and len(results_df) > 0:
        output_dir = Path(config['output_dir'])
        _save_per_tissue_results(results_df, output_dir, ontology_terms)

    logger.info(f"Inference complete. Processed {len(results_df)} regions.")
    
    # Check if multi-track CSV mode
    is_multi_track_csv = config.get('_borzoi_multi_track') is not None
    
    # Validate results - check for multi-track CSV, multi-tissue ATAC, two-track RNA, or single-track
    if len(results_df) > 0:
        if is_multi_track_csv:
            # Multi-track CSV results - report per track
            track_info = config['_borzoi_multi_track']['track_info']
            logger.info(f"Multi-track inference results ({len(track_info)} tracks):")
            
            # Find all pearson_corr columns that match our tracks
            corr_cols = [col for col in results_df.columns if col.startswith('pearson_corr_')]
            for col in sorted(corr_cols):
                mean_corr = results_df[col].mean()
                track_name = col.replace('pearson_corr_', '')
                logger.info(f"  {track_name}: Mean Pearson r = {mean_corr:.3f}")
                
            # Save per-track results and per-sample correlations
            if 'output_dir' in config:
                output_dir = Path(config['output_dir'])
                _save_per_track_results(results_df, output_dir, list(track_info.keys()))
                _save_per_sample_correlations(predictions, output_dir, list(track_info.keys()))
                
        elif is_multi_tissue_atac:
            # Multi-tissue ATAC results - report per tissue
            logger.info("Multi-tissue ATAC inference results:")
            for tissue in ontology_terms:
                tissue_key = tissue.replace(':', '_').replace('-', '_')
                corr_col = f'pearson_corr_{tissue_key}'
                if corr_col in results_df.columns:
                    mean_corr = results_df[corr_col].mean()
                    logger.info(f"  {tissue}: Mean Pearson correlation = {mean_corr:.3f}")
        elif 'pearson_corr_encode' in results_df.columns:
            # Two-track RNA results
            logger.info(f"Mean Pearson correlation (Encode): {results_df['pearson_corr_encode'].mean():.3f}")
            logger.info(f"Mean Pearson correlation (GTEx): {results_df['pearson_corr_gtex'].mean():.3f}")
        elif 'pearson_corr' in results_df.columns:
            # Single-track ATAC results (backward compatible)
            logger.info(f"Mean Pearson correlation: {results_df['pearson_corr'].mean():.3f}")
        else:
            logger.error("No valid results produced. Missing correlation columns.")
            raise ValueError("Inference failed: No valid predictions were generated")
    else:
        logger.error("No regions processed successfully.")
        raise ValueError("Inference failed: No valid predictions were generated")

    # Step 6: Create visualization plots (if requested)
    if config.get('create_plots', True) and 'output_dir' in config:
        logger.info("Step 6/6: Creating visualization plots...")
        plots_dir = Path(config['output_dir']) / 'plots'

        # Determine model name for visualization labels
        model_type = config.get('model_type', 'alphagenome')
        if model_type == 'borzoi':
            model_name = 'Borzoi'
        else:
            model_name = 'AlphaGenome'

        plot_paths = create_inference_plots(
            results_df=results_df,
            predictions_dict=predictions,
            output_dir=plots_dir,
            plot_per_sample=config.get('plot_per_sample', True),
            plot_per_region=config.get('plot_per_region', False),
            plot_formats=config.get('plot_formats', ['png', 'pdf']),
            dpi=config.get('plot_dpi', 150),
            predixcan_results_path=config.get('predixcan_results_path'),
            model_name=model_name
        )

        logger.info(f"Created plots in: {plots_dir}")
        logger.info(f"  - {len(plot_paths['summary'])} summary plots")
        logger.info(f"  - {len(plot_paths['per_sample'])} per-sample plots")
        logger.info(f"  - {len(plot_paths['per_region'])} per-region plots")

    return results_df, predictions


def _validate_config(config: Dict) -> None:
    """Validate required configuration parameters."""
    base_required = ['data_type', 'vcf_file_path', 'hg38_file_path', 'window_size']
    
    # Validate base requirements first so we can safely use them below
    for key in base_required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    required = []
    model_type = config.get('model_type', 'alphagenome')
    data_type = config['data_type']
    
    if model_type == 'alphagenome':
        required.append('api_key')
        required.append('ontology_terms')
    elif model_type == 'borzoi':
        # Determine effective output_type for validation
        output_type = config.get('output_type')
        if output_type is None:
            output_type = 'atac' if data_type == 'peaks' else 'rna'
        
        # ontology_terms is only optional if track config is provided:
        # - borzoi_tracks_csv: CSV file with multiple tracks (preferred)
        # - borzoi_rna_track / borzoi_atac_track: single track identifier
        borzoi_tracks_csv = config.get('borzoi_tracks_csv')
        borzoi_rna_track = config.get('borzoi_rna_track')
        borzoi_atac_track = config.get('borzoi_atac_track')
        
        has_track_config = (
            borzoi_tracks_csv or
            (output_type == 'rna' and borzoi_rna_track) or
            (output_type == 'atac' and borzoi_atac_track)
        )
        
        if not has_track_config:
            required.append('ontology_terms')
    
    # Validate additional requirements
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

    if config.get('save_predictions', False) and 'output_dir' not in config:
        raise ValueError("output_dir required when save_predictions=True")


def _prepare_inference_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[Dict]]:
    """
    Load and prepare data for inference.

    Returns:
        Tuple of (metadata, expr_data, sample_names, sample_lists)
    """
    if config['data_type'] == 'peaks':
        metadata, expr_data, sample_names = load_peaks(
            data_dir=config['data_dir'],
            n_peaks=config.get('n_regions'),
            selection_method=config.get('selection_method', 'variance'),
            predixcan_results_path=config.get('predixcan_results_path'),
            predixcan_metric=config.get('predixcan_metric', 'test_pearson'),
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
            predixcan_results_path=config.get('predixcan_results_path'),
            predixcan_metric=config.get('predixcan_metric', 'test_pearson'),
            random_state=config.get('random_seed', 42),
            start_rank=config.get('region_start_rank'),
            end_rank=config.get('region_end_rank')
        )
        sample_names = list(expr_data.columns)
    
    # Load train/test split if provided
    if config.get('split_file_path'):
        split_data = _load_train_test_split(config['split_file_path'], sample_names)
        if sample_lists is None:
            sample_lists = {}
        sample_lists.update(split_data)
        logger.info(f"Loaded split from {config['split_file_path']}")
        logger.info(f"  Train: {len(split_data['train_samples'])}, Test: {len(split_data['test_samples'])}")

    return metadata, expr_data, sample_names, sample_lists


def _load_train_test_split(split_file_path: Union[str, Path], available_samples: List[str]) -> Dict[str, List[str]]:
    """
    Load train/test split from JSON file.
    
    Args:
        split_file_path: Path to train_test_split.json file
        available_samples: List of available samples to validate against
        
    Returns:
        Dict with 'train_samples', 'test_samples', and optionally 'val_samples'
        
    Raises:
        FileNotFoundError: If split file doesn't exist
        ValueError: If split contains samples not in available_samples
    """
    import json
    
    split_file_path = Path(split_file_path)
    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")
    
    with open(split_file_path, 'r') as f:
        split_data = json.load(f)
    
    train_samples = split_data['train_samples']
    test_samples = split_data['test_samples']
    val_samples = split_data.get('val_samples', [])
    
    # Validate samples exist in data
    available_set = set(available_samples)
    train_set = set(train_samples)
    test_set = set(test_samples)
    val_set = set(val_samples) if val_samples else set()
    
    missing_train = train_set - available_set
    missing_test = test_set - available_set
    missing_val = val_set - available_set
    
    if missing_train or missing_test or missing_val:
        error_msg = "Split contains samples not in current data:\n"
        if missing_train:
            error_msg += f"  Missing from training: {missing_train}\n"
        if missing_test:
            error_msg += f"  Missing from testing: {missing_test}\n"
        if missing_val:
            error_msg += f"  Missing from validation: {missing_val}\n"
        raise ValueError(error_msg)
    
    result = {
        'train_samples': train_samples,
        'test_samples': test_samples
    }
    
    if val_samples:
        result['val_samples'] = val_samples
    
    return result


def _run_predictions(
    metadata: pd.DataFrame,
    expr_data: pd.DataFrame,
    sample_names: List[str],
    sample_lists: Optional[Dict],
    model,
    sequence_length_bp: int,
    config: Dict,
    tracks_df: Optional[pd.DataFrame] = None,
    genome_dataset: Optional['GenomeDataset'] = None
) -> Tuple[List[Dict], Dict]:
    """
    Run predictions across all regions.

    Args:
        genome_dataset: Optional pre-initialized GenomeDataset for efficient
                       sequence extraction. If None, falls back to per-sample
                       get_personal_sequences() calls.

    Returns:
        Tuple of (results_list, predictions_dict)
    """
    enable_parallel = config.get('enable_parallel', True)
    max_workers = config.get('max_workers', 8)

    results_list = []
    predictions_dict = {}

    if enable_parallel and len(metadata) > 1:
        # Parallel processing
        results_list, predictions_dict = _run_predictions_parallel(
            metadata=metadata,
            expr_data=expr_data,
            sample_names=sample_names,
            sample_lists=sample_lists,
            model=model,
            sequence_length_bp=sequence_length_bp,
            config=config,
            max_workers=max_workers,
            tracks_df=tracks_df,
            genome_dataset=genome_dataset
        )
    else:
        # Sequential processing with progress bar
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Predicting"):
            try:
                result, pred_data = _process_single_region(
                    region_data=(idx, row),
                    expr_data=expr_data,
                    sample_names=sample_names,
                    sample_lists=sample_lists,
                    model=model,
                    sequence_length_bp=sequence_length_bp,
                    config=config,
                    tracks_df=tracks_df,
                    genome_dataset=genome_dataset
                )
                results_list.append(result)
                predictions_dict[result['region_id']] = pred_data
            except Exception as e:
                logger.error(f"Failed to process region {idx}: {str(e)}")
                continue

    return results_list, predictions_dict


def _run_predictions_parallel(
    metadata: pd.DataFrame,
    expr_data: pd.DataFrame,
    sample_names: List[str],
    sample_lists: Optional[Dict],
    model,
    sequence_length_bp: int,
    config: Dict,
    max_workers: int,
    tracks_df: Optional[pd.DataFrame] = None,
    genome_dataset: Optional['GenomeDataset'] = None
) -> Tuple[List[Dict], Dict]:
    """Run predictions in parallel using ThreadPoolExecutor."""
    results_list = []
    predictions_dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_region = {
            executor.submit(
                _process_single_region,
                (idx, row),
                expr_data,
                sample_names,
                sample_lists,
                model,
                sequence_length_bp,
                config,
                tracks_df,
                genome_dataset
            ): idx
            for idx, row in metadata.iterrows()
        }

        # Collect results with progress bar
        with tqdm(total=len(future_to_region), desc="Predicting (parallel)") as pbar:
            for future in as_completed(future_to_region):
                region_idx = future_to_region[future]
                try:
                    result, pred_data = future.result()
                    results_list.append(result)
                    predictions_dict[result['region_id']] = pred_data
                except Exception as e:
                    logger.error(f"Failed to process region {region_idx}: {str(e)}")
                finally:
                    pbar.update(1)

    return results_list, predictions_dict


def _process_single_region(
    region_data: Tuple,
    expr_data: pd.DataFrame,
    sample_names: List[str],
    sample_lists: Optional[Dict],
    model,
    sequence_length_bp: int,
    config: Dict,
    tracks_df: Optional[pd.DataFrame] = None,
    genome_dataset: Optional['GenomeDataset'] = None
) -> Tuple[Dict, Dict]:
    """
    Process a single genomic region with two-track RNA support.

    Args:
        genome_dataset: Optional pre-initialized GenomeDataset for efficient
                       sequence extraction. If None, uses get_personal_sequences().

    Returns:
        Tuple of (result_dict, prediction_data)
    """
    idx, region = region_data
    region_id = region.get('ensg', idx)

    # Determine which samples to use
    if sample_lists and config.get('target_sample_set'):
        target_samples = sample_lists[config['target_sample_set']]
    else:
        target_samples = sample_names[:config.get('n_samples', len(sample_names))]

    # Get observed expression
    observed = expr_data.loc[region_id, target_samples].values

    # Run predictions for each sample
    predictions = []
    predictions_encode = []  # For RNA: encode_combined track
    predictions_gtex = []    # For RNA: gtex track
    predictions_per_tissue = {}  # For multi-tissue ATAC: tissue -> [predictions]
    valid_samples = []

    output_type = config.get('output_type', 'atac' if config['data_type'] == 'peaks' else 'rna')
    model_type = config.get('model_type', 'alphagenome')
    
    # Check if multi-tissue ATAC mode
    ontology_terms = config.get('ontology_terms', [])
    is_multi_tissue_atac = (output_type == 'atac' and len(ontology_terms) > 1)
    
    # For Borzoi, we need to identify track indices for each tissue/assay
    borzoi_track_indices = {}
    borzoi_rna_paired_strand = None  # For paired strand RNA prediction (ENCFF196HWN style)
    borzoi_multi_track_info = None   # For multi-track CSV mode
    predictions_per_track = {}       # For multi-track: track_id -> [predictions]
    
    if model_type == 'borzoi' and tracks_df is not None:
        # Check for multi-track CSV mode first (takes precedence)
        if '_borzoi_multi_track' in config:
            borzoi_multi_track_info = config['_borzoi_multi_track']
            # Initialize prediction lists for each track
            for track_id in borzoi_multi_track_info['track_info']:
                predictions_per_track[track_id] = []
        
        # Check for borzoi_rna_track config (e.g., ENCFF196HWN for ROSMAP brain)
        borzoi_rna_track = config.get('borzoi_rna_track')
        borzoi_atac_track = config.get('borzoi_atac_track')
        
        if borzoi_rna_track and output_type == 'rna':
            # Use paired strand prediction with specific track identifier
            try:
                plus_idx, minus_idx = get_borzoi_rna_track_by_identifier(tracks_df, borzoi_rna_track)
                borzoi_rna_paired_strand = (plus_idx, minus_idx)
                logger.info(f"Using Borzoi RNA track {borzoi_rna_track} (+:{plus_idx}, -:{minus_idx})")
            except ValueError as e:
                logger.error(f"Failed to find Borzoi RNA track: {e}")
                raise
        elif borzoi_atac_track and output_type == 'atac':
            # Use specific ATAC/DNase track identifier (e.g., ENCFF915DFR for LCL)
            try:
                # For ATAC tracks, there's typically a single track (not paired strands)
                # Look up track by identifier
                track_match = tracks_df[tracks_df['identifier'] == borzoi_atac_track]
                if len(track_match) == 0:
                    raise ValueError(f"Could not find ATAC track for identifier: {borzoi_atac_track}")
                
                track_idx = int(track_match.index[0])
                # Store as a single-element list under a dummy key for compatibility
                borzoi_track_indices['_atac_track'] = [track_idx]
                logger.info(f"Using Borzoi ATAC track {borzoi_atac_track} (index: {track_idx})")
            except ValueError as e:
                logger.error(f"Failed to find Borzoi ATAC track: {e}")
                raise
            
            # Guard against misconfiguration: borzoi_atac_track with multiple ontology_terms
            if is_multi_tissue_atac:
                logger.warning(
                    f"borzoi_atac_track is set with multiple ontology_terms ({len(ontology_terms)}). "
                    f"Using single-tissue mode with the specified track '{borzoi_atac_track}'. "
                    f"Multi-tissue ATAC requires ontology-based track selection, not a specific track."
                )
                is_multi_tissue_atac = False
        else:
            # Fallback to ontology-based track selection
            for term in ontology_terms:
                # Map ontology term to Borzoi track description query
                # This mapping needs to be provided or inferred. 
                # For now, we assume the ontology term IS the query or provided in a mapping
                query = term  # In real usage, might need a mapping dict in config
                if 'borzoi_mapping' in config:
                    query = config['borzoi_mapping'].get(term, term)
                    
                indices = get_borzoi_track_indices(tracks_df, query, output_type.upper())
                if not indices:
                    logger.warning(f"No Borzoi tracks found for {term} ({output_type})")
                borzoi_track_indices[term] = indices
    
    # Initialize per-tissue prediction lists if multi-tissue mode
    if is_multi_tissue_atac:
        for tissue in ontology_terms:
            predictions_per_tissue[tissue] = []

    # Create sample ID mapping if using genome_dataset
    sample_id_to_idx = None
    gene_idx = None
    if genome_dataset is not None:
        sample_id_to_idx = {sid: i for i, sid in enumerate(genome_dataset.sample_list)}
        # Get gene index: use ENSG ID to find position in GenomeDataset
        # The GenomeDataset has reset_index(drop=True) so it uses positional indices 0,1,2,...
        region_ensg = region.get('ensg', region_id)
        try:
            # Find which row in genome_dataset.gene_metadata has this ENSG
            gene_idx = genome_dataset.gene_metadata[genome_dataset.gene_metadata['ensg'] == region_ensg].index[0]
        except (IndexError, KeyError):
            logger.warning(f"Gene {region_ensg} not found in genome_dataset, falling back to old method")
            gene_idx = None
            sample_id_to_idx = None

    for sample_idx, sample_id in enumerate(target_samples):
        try:
            # Initialize transformers to None (for fallback path)
            mat_transformer = None
            pat_transformer = None
            
            # Extract sequences using dataset or fallback to utility function
            if genome_dataset is not None and gene_idx is not None and sample_id in sample_id_to_idx:
                # Use shared dataset: O(1) cache lookup after first sample
                dataset_sample_idx = sample_id_to_idx[sample_id]
                dataset_idx = gene_idx * genome_dataset.n_samples + dataset_sample_idx
                dataset_output = genome_dataset[dataset_idx]
                
                # Extract sequences and transformers
                if len(dataset_output) >= 3:
                    seq_arr, _, transformers = dataset_output[:3]
                    mat_transformer, pat_transformer = transformers
                else:
                    seq_arr = dataset_output[0]
                
                # seq_arr shape: (2, 2) where seq_arr[1] = (maternal, paternal)
                maternal_seq, paternal_seq = seq_arr[1]
            else:
                # Fallback: old behavior (for backward compatibility / non-Borzoi)
                window_size = config['window_size']
                half_window = window_size // 2
                start = region['tss'] - half_window
                end = region['tss'] + half_window

                maternal_seq, paternal_seq = get_personal_sequences(
                    vcf_path=config['vcf_file_path'],
                    genome_path=config['hg38_file_path'],
                    chrom=region['chr'],
                    start=start,
                    end=end,
                    sample_id=sample_id,
                    contig_prefix=config.get('contig_prefix', '')
                )

            if model_type == 'borzoi':
                # Borzoi Prediction Logic
                
                # Get region boundaries for aggregation
                if output_type == 'atac':
                    region_start = region['Pos_Left']
                    region_end = region['Pos_Right']
                else:
                    region_start = region['start']
                    region_end = region['end']
                
                seq_center = region['tss']
                
                # Multi-track CSV mode: use predict_borzoi_multi_track
                if borzoi_multi_track_info is not None:
                    track_preds = predict_borzoi_multi_track(
                        model=model,
                        maternal_sequence=maternal_seq,
                        paternal_sequence=paternal_seq,
                        all_indices=borzoi_multi_track_info['all_indices'],
                        track_info=borzoi_multi_track_info['track_info'],
                        output_type=borzoi_multi_track_info['output_type'],
                        region_start=region_start,
                        region_end=region_end,
                        seq_center=seq_center,
                        device=config.get('_borzoi_device', 'cuda'),
                        maternal_transformer=mat_transformer,
                        paternal_transformer=pat_transformer
                    )

                    # Only keep this sample if *all* tracks produced valid predictions.
                    # This keeps per-track prediction lists aligned with valid_samples/observed_valid.
                    if not track_preds:
                        # No predictions returned for this sample; skip it entirely.
                        continue

                    if any(np.isnan(pred_val) for pred_val in track_preds.values()):
                        # At least one track failed; do NOT append anything for this sample.
                        continue

                    # All tracks valid: append predictions for every track and mark sample as valid.
                    for track_id, pred_val in track_preds.items():
                        predictions_per_track[track_id].append(pred_val)

                    valid_samples.append(sample_id)

                    continue  # Skip other Borzoi logic
                
                if output_type == 'rna' and borzoi_rna_paired_strand is not None:
                    # Use paired strand RNA prediction (ENCFF196HWN style)
                    # This is the preferred method for ROSMAP brain expression
                    plus_idx, minus_idx = borzoi_rna_paired_strand
                    
                    val = predict_borzoi_personal_rna_paired_strand(
                        model=model,
                        maternal_sequence=maternal_seq,
                        paternal_sequence=paternal_seq,
                        plus_idx=plus_idx,
                        minus_idx=minus_idx,
                        gene_start=region_start,
                        gene_end=region_end,
                        tss=seq_center,
                        device=config.get('_borzoi_device', 'cuda'),
                        maternal_transformer=mat_transformer,
                        paternal_transformer=pat_transformer
                    )
                    
                    if not np.isnan(val):
                        # Use single prediction value for Borzoi
                        # (stored in predictions list, not encode/gtex split)
                        predictions.append(val)
                        valid_samples.append(sample_id)
                else:
                    # Fallback: use ontology-based track selection
                    # Collect all track indices needed
                    all_indices = []
                    for idx_list in borzoi_track_indices.values():
                        all_indices.extend(idx_list)
                    all_indices = sorted(list(set(all_indices)))
                    
                    if not all_indices:
                        continue
                    
                    # Use predict_borzoi_personal_genome with aggregation
                    # This handles haplotype averaging and region aggregation correctly
                    agg_pred = predict_borzoi_personal_genome(
                        model=model,
                        maternal_sequence=maternal_seq,
                        paternal_sequence=paternal_seq,
                        target_tracks=all_indices,
                        device=config.get('_borzoi_device', 'cuda'),
                        region_start=region_start,
                        region_end=region_end,
                        seq_center=seq_center,
                        aggregate_method='sum',
                        log_transform=True,
                        maternal_transformer=mat_transformer,
                        paternal_transformer=pat_transformer
                    )
                    
                    # Map predictions back to tissues/tracks
                    if output_type == 'atac':
                        if is_multi_tissue_atac:
                            all_valid = True
                            for tissue in ontology_terms:
                                tissue_indices = borzoi_track_indices.get(tissue, [])
                                if tissue_indices:
                                    # Find positions in agg_pred array
                                    pos = [all_indices.index(i) for i in tissue_indices]
                                    # Average over tracks for this tissue
                                    val = float(np.mean(agg_pred[pos]))
                                    predictions_per_tissue[tissue].append(val)
                                else:
                                    all_valid = False
                                    break
                            if all_valid:
                                valid_samples.append(sample_id)
                        else:
                            # Single tissue ATAC
                            # Check if using borzoi_atac_track config (direct track specification)
                            if '_atac_track' in borzoi_track_indices:
                                tissue_indices = borzoi_track_indices['_atac_track']
                            else:
                                # Fallback to ontology-based track selection
                                tissue_indices = borzoi_track_indices.get(ontology_terms[0], []) if ontology_terms else []
                            
                            if tissue_indices:
                                pos = [all_indices.index(i) for i in tissue_indices]
                                val = float(np.mean(agg_pred[pos]))
                                predictions.append(val)
                                valid_samples.append(sample_id)
                                
                    else:  # RNA fallback (without paired strand)
                        # For RNA, average all brain tracks for the tissue
                        tissue_indices = borzoi_track_indices.get(ontology_terms[0], []) if ontology_terms else []
                        if tissue_indices:
                            pos = [all_indices.index(i) for i in tissue_indices]
                            val = float(np.mean(agg_pred[pos]))
                            # Use same value for both encode and gtex slots
                            # (Borzoi doesn't distinguish these like AlphaGenome)
                            predictions_encode.append(val)
                            predictions_gtex.append(val)
                            valid_samples.append(sample_id)

                continue  # Skip AlphaGenome logic

            # Predict (AlphaGenome Logic)
            if output_type == 'atac':
                # For ATAC: use actual peak boundaries from metadata
                peak_start = region['Pos_Left']
                peak_end = region['Pos_Right']
                
                if is_multi_tissue_atac:
                    # Multi-tissue ATAC prediction
                    from alphagenome_eval.utils.prediction import predict_personal_genome_atac_multi_tissue
                    
                    pred_dict = predict_personal_genome_atac_multi_tissue(
                        model=model,
                        maternal_sequence=maternal_seq,
                        paternal_sequence=paternal_seq,
                        ontology_terms=ontology_terms,
                        peak_start=peak_start,
                        peak_end=peak_end,
                        sequence_length_bp=sequence_length_bp,
                        aggregate='sum',
                        maternal_transformer=mat_transformer,
                        paternal_transformer=pat_transformer
                    )
                    
                    # Store per-tissue predictions
                    all_valid = True
                    for tissue in ontology_terms:
                        pred_value = pred_dict.get(tissue)
                        if pred_value is not None:
                            predictions_per_tissue[tissue].append(pred_value)
                        else:
                            all_valid = False
                            break
                    
                    # Only add to valid_samples if all tissues succeeded
                    if all_valid:
                        valid_samples.append(sample_id)
                else:
                    # Single-tissue ATAC (original behavior)
                    pred = predict_personal_genome_atac(
                        model=model,
                        maternal_sequence=maternal_seq,
                        paternal_sequence=paternal_seq,
                        ontology_terms=ontology_terms,
                        peak_start=peak_start,
                        peak_end=peak_end,
                        sequence_length_bp=sequence_length_bp,
                        aggregate='sum',
                        maternal_transformer=mat_transformer,
                        paternal_transformer=pat_transformer
                    )
                    # For ATAC, pred is a single value
                    if pred is not None:
                        predictions.append(pred)
                        valid_samples.append(sample_id)
                    
            else:  # rna - returns dict with two tracks
                # For RNA: use actual gene boundaries from metadata (start_hg38/end_hg38)
                gene_start = region['start']
                gene_end = region['end']
                gene_tss = region['tss']
                
                pred_dict = predict_personal_genome_rna(
                    model=model,
                    maternal_sequence=maternal_seq,
                    paternal_sequence=paternal_seq,
                    ontology_terms=ontology_terms,
                    gene_start=gene_start,
                    gene_end=gene_end,
                    tss=gene_tss,
                    sequence_length_bp=sequence_length_bp,
                    aggregate='sum',
                    maternal_transformer=mat_transformer,
                    paternal_transformer=pat_transformer
                )
                # For RNA, pred_dict is {'encode_combined': value, 'gtex': value}
                if pred_dict is not None:
                    predictions_encode.append(pred_dict['encode_combined'])
                    predictions_gtex.append(pred_dict['gtex'])
                    valid_samples.append(sample_id)
                else:
                    logger.debug(f"Prediction returned None for {region_id}, sample {sample_id}")

        except Exception as e:
            logger.debug(f"Failed prediction for {region_id}, sample {sample_id}: {str(e)}")
            continue

    # Calculate correlation based on output type
    observed_valid = expr_data.loc[region_id, valid_samples].values

    # Check for multi-track CSV mode first
    if predictions_per_track and any(len(preds) > 0 for preds in predictions_per_track.values()):
        # Multi-track CSV mode: calculate correlation per track
        result = {
            'region_id': region_id,
            'region_name': region.get('gene_name', str(region_id)),
            'chr': region['chr'],
            'start': region.get('start', region.get('Pos_Left')),
            'end': region.get('end', region.get('Pos_Right')),
            'tss': region['tss'],
            'n_samples': len(valid_samples),
            'mean_obs': np.mean(observed_valid) if len(observed_valid) > 0 else np.nan,
            'std_obs': np.std(observed_valid) if len(observed_valid) > 0 else np.nan
        }
        
        # Add per-track correlations and statistics
        for track_id, preds in predictions_per_track.items():
            preds_arr = np.array(preds)
            
            if len(preds_arr) >= 3 and len(preds_arr) == len(observed_valid):
                corr, pval = pearsonr(preds_arr, observed_valid)
            else:
                corr, pval = np.nan, np.nan
            
            # Use track_id as column suffix (sanitize for column names)
            track_key = track_id.replace(':', '_').replace('-', '_').replace('+', '_plus').replace(' ', '_')
            result[f'pearson_corr_{track_key}'] = corr
            result[f'p_value_{track_key}'] = pval
            result[f'mean_pred_{track_key}'] = np.mean(preds_arr) if len(preds_arr) > 0 else np.nan
            result[f'std_pred_{track_key}'] = np.std(preds_arr) if len(preds_arr) > 0 else np.nan
        
        pred_data = {
            'predictions_per_track': {k: np.array(v) for k, v in predictions_per_track.items()},
            'observed': observed_valid,
            'sample_ids': valid_samples,
            'track_list': list(predictions_per_track.keys())
        }
        
        return result, pred_data

    if output_type == 'atac':
        if is_multi_tissue_atac:
            # Multi-tissue ATAC: calculate correlation per tissue
            result = {
                'region_id': region_id,
                'region_name': region.get('gene_name', str(region_id)),
                'chr': region['chr'],
                'start': region['Pos_Left'],
                'end': region['Pos_Right'],
                'tss': region['tss'],
                'n_samples': len(valid_samples),
                'mean_obs': np.mean(observed_valid),
                'std_obs': np.std(observed_valid)
            }
            
            # Add per-tissue correlations and statistics
            for tissue in ontology_terms:
                tissue_key = tissue.replace(':', '_').replace('-', '_')
                preds = np.array(predictions_per_tissue[tissue])
                
                if len(preds) >= 3:
                    corr, pval = pearsonr(preds, observed_valid)
                else:
                    corr, pval = np.nan, np.nan
                
                result[f'pearson_corr_{tissue_key}'] = corr
                result[f'p_value_{tissue_key}'] = pval
                result[f'mean_pred_{tissue_key}'] = np.mean(preds) if len(preds) > 0 else np.nan
                result[f'std_pred_{tissue_key}'] = np.std(preds) if len(preds) > 0 else np.nan
            
            pred_data = {
                'predictions_per_tissue': predictions_per_tissue,
                'observed': observed_valid,
                'sample_ids': valid_samples,
                'tissue_list': ontology_terms
            }
        else:
            # Single-tissue ATAC: backward compatible
            predictions = np.array(predictions)
            if len(predictions) >= 3:
                corr, pval = pearsonr(predictions, observed_valid)
            else:
                corr, pval = np.nan, np.nan

            result = {
                'region_id': region_id,
                'region_name': region.get('gene_name', str(region_id)),
                'chr': region['chr'],
                'start': region['Pos_Left'],
                'end': region['Pos_Right'],
                'tss': region['tss'],
                'n_samples': len(predictions),
                'pearson_corr': corr,
                'p_value': pval,
                'mean_pred': np.mean(predictions),
                'std_pred': np.std(predictions),
                'mean_obs': np.mean(observed_valid),
                'std_obs': np.std(observed_valid)
            }

            pred_data = {
                'predictions': predictions,
                'observed': observed_valid,
                'sample_ids': valid_samples
            }

    else:  # RNA
        # Check if using Borzoi paired strand (single prediction value) or AlphaGenome (two tracks)
        if len(predictions) > 0 and len(predictions_encode) == 0:
            # Borzoi paired strand RNA: single prediction value (like ATAC)
            predictions = np.array(predictions)
            if len(predictions) >= 3:
                corr, pval = pearsonr(predictions, observed_valid)
            else:
                corr, pval = np.nan, np.nan

            result = {
                'region_id': region_id,
                'region_name': region.get('gene_name', str(region_id)),
                'chr': region['chr'],
                'start': region['start'],
                'end': region['end'],
                'tss': region['tss'],
                'n_samples': len(predictions),
                'pearson_corr': corr,
                'p_value': pval,
                'mean_pred': np.mean(predictions),
                'std_pred': np.std(predictions),
                'mean_obs': np.mean(observed_valid),
                'std_obs': np.std(observed_valid)
            }

            pred_data = {
                'predictions': predictions,
                'observed': observed_valid,
                'sample_ids': valid_samples
            }
        else:
            # AlphaGenome RNA: two-track correlation (encode + gtex)
            predictions_encode = np.array(predictions_encode)
            predictions_gtex = np.array(predictions_gtex)
            
            if len(predictions_encode) >= 3:
                corr_encode, pval_encode = pearsonr(predictions_encode, observed_valid)
                corr_gtex, pval_gtex = pearsonr(predictions_gtex, observed_valid)
            else:
                corr_encode, pval_encode = np.nan, np.nan
                corr_gtex, pval_gtex = np.nan, np.nan

            result = {
                'region_id': region_id,
                'region_name': region.get('gene_name', str(region_id)),
                'chr': region['chr'],
                'start': region['start'],
                'end': region['end'],
                'tss': region['tss'],
                'n_samples': len(predictions_encode),
                
                # Encode combined track (sum of + and - strands)
                'pearson_corr_encode': corr_encode,
                'p_value_encode': pval_encode,
                'mean_pred_encode': np.mean(predictions_encode),
                'std_pred_encode': np.std(predictions_encode),
                
                # GTEx track
                'pearson_corr_gtex': corr_gtex,
                'p_value_gtex': pval_gtex,
                'mean_pred_gtex': np.mean(predictions_gtex),
                'std_pred_gtex': np.std(predictions_gtex),
                
                # Observed data
                'mean_obs': np.mean(observed_valid),
                'std_obs': np.std(observed_valid)
            }

            pred_data = {
                'predictions_encode': predictions_encode,
                'predictions_gtex': predictions_gtex,
                'observed': observed_valid,
                'sample_ids': valid_samples
            }

    return result, pred_data


def _save_predictions(predictions: Dict, output_dir: str) -> None:
    """Save prediction results to disk (supports ATAC single/multi-tissue, RNA, and multi-track formats)."""
    # Create predictions subfolder
    output_path = Path(output_dir) / 'predictions'
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as npz for each region
    for region_id, data in predictions.items():
        save_path = output_path / f"{region_id}_predictions.npz"
        
        # Check format type
        if 'predictions_per_track' in data:
            # Multi-track CSV format
            save_dict = {}
            for track_id, preds in data['predictions_per_track'].items():
                track_key = track_id.replace(':', '_').replace('-', '_').replace('+', '_plus').replace(' ', '_')
                save_dict[f'predictions_{track_key}'] = np.array(preds)
            
            save_dict['observed'] = data['observed']
            save_dict['sample_ids'] = data['sample_ids']
            save_dict['track_list'] = np.array(data['track_list'], dtype=str)
            
            np.savez(save_path, **save_dict)
            
        elif 'predictions_per_tissue' in data:
            # Multi-tissue ATAC format
            save_dict = {}
            for tissue, preds in data['predictions_per_tissue'].items():
                tissue_key = tissue.replace(':', '_').replace('-', '_')
                save_dict[f'predictions_{tissue_key}'] = np.array(preds)
            
            save_dict['observed'] = data['observed']
            save_dict['sample_ids'] = data['sample_ids']
            save_dict['tissue_list'] = np.array(data['tissue_list'], dtype=str)
            
            np.savez(save_path, **save_dict)
            
        elif 'predictions_encode' in data:
            # RNA format with two tracks
            np.savez(
                save_path,
                predictions_encode=data['predictions_encode'],
                predictions_gtex=data['predictions_gtex'],
                observed=data['observed'],
                sample_ids=data['sample_ids']
            )
        else:
            # Single-tissue ATAC format (backward compatible)
            np.savez(
                save_path,
                predictions=data['predictions'],
                observed=data['observed'],
                sample_ids=data['sample_ids']
            )

    logger.info(f"Saved {len(predictions)} NPZ prediction files to {output_path}")


def _save_per_tissue_results(results_df: pd.DataFrame, output_dir: Path, ontology_terms: List[str]) -> None:
    """
    Save separate CSV files for each tissue in multi-tissue mode.
    
    Args:
        results_df: DataFrame with all results (contains per-tissue columns)
        output_dir: Output directory
        ontology_terms: List of tissue ontology terms
    """
    per_tissue_dir = output_dir / 'per_tissue'
    per_tissue_dir.mkdir(parents=True, exist_ok=True)
    
    # Base columns (common to all tissues)
    base_cols = ['region_id', 'region_name', 'chr', 'start', 'end', 'tss', 'n_samples', 'mean_obs', 'std_obs']
    
    tissue_summary = []
    
    for tissue in ontology_terms:
        tissue_key = tissue.replace(':', '_').replace('-', '_')
        
        # Find columns for this tissue
        tissue_specific_cols = [
            f'pearson_corr_{tissue_key}',
            f'p_value_{tissue_key}',
            f'mean_pred_{tissue_key}',
            f'std_pred_{tissue_key}'
        ]
        
        # Check if columns exist
        existing_tissue_cols = [c for c in tissue_specific_cols if c in results_df.columns]
        
        if existing_tissue_cols:
            # Select columns for this tissue
            cols_to_save = base_cols + existing_tissue_cols
            tissue_df = results_df[cols_to_save].copy()
            
            # Rename columns to remove tissue suffix for cleaner CSVs
            rename_map = {
                f'pearson_corr_{tissue_key}': 'pearson_corr',
                f'p_value_{tissue_key}': 'p_value',
                f'mean_pred_{tissue_key}': 'mean_pred',
                f'std_pred_{tissue_key}': 'std_pred'
            }
            tissue_df.rename(columns=rename_map, inplace=True)
            
            # Add tissue identifier column
            tissue_df.insert(0, 'tissue_ontology', tissue)
            
            # Save to CSV
            filename = f"{tissue_key}_results.csv"
            tissue_df.to_csv(per_tissue_dir / filename, index=False)
            
            # Calculate summary statistics
            valid_corr = tissue_df['pearson_corr'].dropna()
            if len(valid_corr) > 0:
                tissue_summary.append({
                    'tissue_ontology': tissue,
                    'n_regions': len(tissue_df),
                    'n_valid_correlations': len(valid_corr),
                    'mean_correlation': valid_corr.mean(),
                    'median_correlation': valid_corr.median(),
                    'std_correlation': valid_corr.std(),
                    'min_correlation': valid_corr.min(),
                    'max_correlation': valid_corr.max()
                })
    
    # Save tissue summary
    if tissue_summary:
        summary_df = pd.DataFrame(tissue_summary)
        summary_df.to_csv(output_dir / 'tissue_summary.csv', index=False)
        logger.info(f"Saved per-tissue results to {per_tissue_dir}")
        logger.info(f"Saved tissue summary to {output_dir / 'tissue_summary.csv'}")


def _save_per_track_results(results_df: pd.DataFrame, output_dir: Path, track_ids: List[str]) -> None:
    """
    Save separate CSV files for each track in multi-track CSV mode.
    
    Args:
        results_df: DataFrame with all results (contains per-track columns)
        output_dir: Output directory
        track_ids: List of track identifiers
    """
    per_track_dir = output_dir / 'per_track'
    per_track_dir.mkdir(parents=True, exist_ok=True)
    
    # Base columns (common to all tracks)
    base_cols = ['region_id', 'region_name', 'chr', 'start', 'end', 'tss', 'n_samples', 'mean_obs', 'std_obs']
    
    track_summary = []
    
    for track_id in track_ids:
        # Sanitize track_id for column matching
        track_key = track_id.replace(':', '_').replace('-', '_').replace('+', '_plus').replace(' ', '_')
        
        # Find columns for this track
        track_specific_cols = [
            f'pearson_corr_{track_key}',
            f'p_value_{track_key}',
            f'mean_pred_{track_key}',
            f'std_pred_{track_key}'
        ]
        
        # Check if columns exist
        existing_track_cols = [c for c in track_specific_cols if c in results_df.columns]
        
        if existing_track_cols:
            # Select columns for this track
            cols_to_save = [c for c in base_cols if c in results_df.columns] + existing_track_cols
            track_df = results_df[cols_to_save].copy()
            
            # Rename columns to remove track suffix for cleaner CSVs
            rename_map = {
                f'pearson_corr_{track_key}': 'pearson_corr',
                f'p_value_{track_key}': 'p_value',
                f'mean_pred_{track_key}': 'mean_pred',
                f'std_pred_{track_key}': 'std_pred'
            }
            track_df.rename(columns=rename_map, inplace=True)
            
            # Add track identifier column
            track_df.insert(0, 'track_id', track_id)
            
            # Save to CSV
            filename = f"{track_key}_results.csv"
            track_df.to_csv(per_track_dir / filename, index=False)
            
            # Calculate summary statistics
            valid_corr = track_df['pearson_corr'].dropna()
            if len(valid_corr) > 0:
                track_summary.append({
                    'track_id': track_id,
                    'n_regions': len(track_df),
                    'n_valid_correlations': len(valid_corr),
                    'mean_correlation': valid_corr.mean(),
                    'median_correlation': valid_corr.median(),
                    'std_correlation': valid_corr.std(),
                    'min_correlation': valid_corr.min(),
                    'max_correlation': valid_corr.max()
                })
    
    # Save track summary
    if track_summary:
        summary_df = pd.DataFrame(track_summary)
        # Sort by mean correlation descending
        summary_df = summary_df.sort_values('mean_correlation', ascending=False)
        summary_df.to_csv(output_dir / 'track_summary.csv', index=False)
        logger.info(f"Saved per-track results to {per_track_dir}")
        logger.info(f"Saved track summary to {output_dir / 'track_summary.csv'}")


def _save_per_sample_correlations(
    predictions_dict: Dict[str, Dict],
    output_dir: Path,
    track_ids: List[str]
) -> None:
    """
    Save per-sample correlation CSV for multi-track mode.
    
    Creates a wide-format CSV with one row per sample and one column per track,
    where each cell contains the Pearson correlation (predicted vs. observed
    across all regions) for that sample-track combination.
    
    Args:
        predictions_dict: Dict mapping region_id -> {
            'predictions_per_track': {track_id: array of predictions},
            'observed': array of observed values,
            'sample_ids': list of sample IDs
        }
        output_dir: Output directory
        track_ids: List of track identifiers
    """
    from scipy.stats import pearsonr
    
    # Reorganize data by sample
    # sample_data[sample_id][track_id] = {'predicted': [], 'observed': []}
    sample_data: Dict[str, Dict[str, Dict[str, List]]] = {}
    
    for region_id, region_data in predictions_dict.items():
        if 'predictions_per_track' not in region_data:
            continue
            
        predictions_per_track = region_data['predictions_per_track']
        observed = region_data['observed']
        sample_ids = region_data['sample_ids']
        
        for i, sample_id in enumerate(sample_ids):
            if sample_id not in sample_data:
                sample_data[sample_id] = {track_id: {'predicted': [], 'observed': []} for track_id in track_ids}
            
            obs_val = observed[i]
            
            for track_id in track_ids:
                if track_id in predictions_per_track:
                    preds = predictions_per_track[track_id]
                    if i < len(preds):
                        sample_data[sample_id][track_id]['predicted'].append(preds[i])
                        sample_data[sample_id][track_id]['observed'].append(obs_val)
    
    # Calculate correlations for each sample-track pair
    rows = []
    for sample_id in sorted(sample_data.keys()):
        row = {'sample_id': sample_id}
        
        for track_id in track_ids:
            track_key = track_id.replace(':', '_').replace('-', '_').replace('+', '_plus').replace(' ', '_')
            
            pred_list = sample_data[sample_id][track_id]['predicted']
            obs_list = sample_data[sample_id][track_id]['observed']
            
            if len(pred_list) >= 3:
                pred_arr = np.array(pred_list)
                obs_arr = np.array(obs_list)
                # Remove any NaN values
                valid_mask = ~(np.isnan(pred_arr) | np.isnan(obs_arr))
                if valid_mask.sum() >= 3:
                    corr, _ = pearsonr(pred_arr[valid_mask], obs_arr[valid_mask])
                else:
                    corr = np.nan
            else:
                corr = np.nan
            
            row[f'pearson_corr_{track_key}'] = corr
        
        rows.append(row)
    
    # Create DataFrame and save
    if rows:
        sample_df = pd.DataFrame(rows)
        sample_df.to_csv(output_dir / 'sample_correlations.csv', index=False)
        logger.info(f"Saved per-sample correlations to {output_dir / 'sample_correlations.csv'}")

