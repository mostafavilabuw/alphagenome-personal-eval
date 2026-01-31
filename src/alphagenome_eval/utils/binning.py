"""
Binning utilities for ATAC-seq signal aggregation.

This module provides functions to convert high-resolution ATAC predictions
to fixed-size bins for use as features in PrediXcan-style models.

Key Functions:
    - bin_atac_track: Bin ATAC track from source resolution to target bin size
    - bin_atac_around_tss: Extract and bin ATAC signal in a window around TSS
    - extract_binned_features_borzoi: Extract binned features using Borzoi model
    - extract_binned_features_alphagenome: Extract binned features using AlphaGenome

Resolution Notes:
    - AlphaGenome: 1 bp resolution (variable output length based on input)
    - Borzoi: 32 bp per bin (6144 bins covering 196,608 bp centered on 524,288 bp input)
    - Default target: 256 bp bins (8 Borzoi bins each) for clean integer ratio

Author: AlphaGenome Evaluation Team
"""

import numpy as np
from typing import Literal, Optional, Tuple, Dict, List, Any, Union, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd

if TYPE_CHECKING:
    from alphagenome_eval.utils.coordinates import CoordinateTransformer

logger = logging.getLogger(__name__)

# Constants
ALPHAGENOME_RESOLUTION = 1  # 1 bp resolution
BORZOI_RESOLUTION = 32      # 32 bp per bin

# Default binning parameters
DEFAULT_BIN_SIZE = 256      # 256 bp bins (8 Borzoi bins each)
DEFAULT_WINDOW_SIZE = 100000  # 100 KB window around TSS


def bin_atac_track(
    track: np.ndarray,
    source_resolution: int,
    target_bin_size: int = DEFAULT_BIN_SIZE,
    aggregation: Literal['mean', 'sum', 'max'] = 'mean'
) -> np.ndarray:
    """
    Bin ATAC track from source resolution to target bin size.

    Args:
        track: 1D array of ATAC values at source resolution.
               Shape: (n_source_bins,) or (n_positions,)
        source_resolution: Resolution of input in bp
                          - AlphaGenome: 1 (1bp per position)
                          - Borzoi: 32 (32bp per bin)
        target_bin_size: Target bin size in bp (default: 256)
        aggregation: Aggregation method ('mean', 'sum', 'max')

    Returns:
        Binned track of shape (n_target_bins,)

    Example:
        >>> # Borzoi: 6144 bins at 32bp -> bins at 256bp
        >>> borzoi_track = np.random.rand(6144)
        >>> binned = bin_atac_track(borzoi_track, source_resolution=32, target_bin_size=256)
        >>> print(binned.shape)  # Each 256bp bin = 8 source bins

        >>> # AlphaGenome: 100000bp at 1bp -> bins at 256bp
        >>> ag_track = np.random.rand(100000)
        >>> binned = bin_atac_track(ag_track, source_resolution=1, target_bin_size=256)
        >>> print(binned.shape)  # (390,) for 100KB/256bp

    Notes:
        - For Borzoi (32bp bins) to 256bp: each target bin = 8 source bins (clean ratio)
        - For AlphaGenome (1bp) to 256bp: each target bin = 256 source positions
        - Handles edge cases where source doesn't evenly divide into target
    """
    if len(track) == 0:
        return np.array([])

    # Calculate source positions per target bin
    source_positions_per_target_bin = target_bin_size // source_resolution

    if source_positions_per_target_bin < 1:
        raise ValueError(
            f"Target bin size ({target_bin_size}) must be >= source resolution ({source_resolution})"
        )

    n_source = len(track)
    n_target_bins = n_source // source_positions_per_target_bin

    if n_target_bins == 0:
        # Track is shorter than one target bin, aggregate all
        if aggregation == 'mean':
            return np.array([np.mean(track)])
        elif aggregation == 'sum':
            return np.array([np.sum(track)])
        else:  # max
            return np.array([np.max(track)])

    # Reshape for aggregation (truncate to fit evenly)
    usable_length = n_target_bins * source_positions_per_target_bin
    reshaped = track[:usable_length].reshape(n_target_bins, source_positions_per_target_bin)

    # Aggregate
    if aggregation == 'mean':
        binned = np.mean(reshaped, axis=1)
    elif aggregation == 'sum':
        binned = np.sum(reshaped, axis=1)
    elif aggregation == 'max':
        binned = np.max(reshaped, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return binned


def bin_atac_around_tss(
    track: np.ndarray,
    source_resolution: int,
    source_center_idx: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_bin_size: int = DEFAULT_BIN_SIZE,
    aggregation: Literal['mean', 'sum', 'max'] = 'mean'
) -> np.ndarray:
    """
    Extract and bin ATAC signal in a window around TSS.

    Args:
        track: Full ATAC prediction track
        source_resolution: Resolution of input in bp
        source_center_idx: Index of TSS in the track
        window_size: Total window size in bp (default: 100KB)
        target_bin_size: Target bin size in bp (default: 256)
        aggregation: Aggregation method

    Returns:
        Binned track of shape (window_size // target_bin_size,)
        For 100KB window with 256bp bins: shape = (390,)

    Example:
        >>> # Borzoi track centered on TSS
        >>> track = np.random.rand(6144)  # 196KB coverage
        >>> center_idx = 6144 // 2  # TSS at center
        >>> features = bin_atac_around_tss(
        ...     track, source_resolution=32, source_center_idx=center_idx,
        ...     window_size=100000, target_bin_size=256
        ... )
        >>> print(features.shape)  # (390,)
    """
    # Calculate half window in source units
    half_window_bp = window_size // 2
    half_window_source = half_window_bp // source_resolution

    # Calculate start and end indices
    start_idx = source_center_idx - half_window_source
    end_idx = source_center_idx + half_window_source

    # Handle boundary cases with padding
    n_source = len(track)
    pad_left = 0
    pad_right = 0

    if start_idx < 0:
        pad_left = -start_idx
        start_idx = 0

    if end_idx > n_source:
        pad_right = end_idx - n_source
        end_idx = n_source

    # Extract window
    window = track[start_idx:end_idx]

    # Pad if needed (with zeros for regions outside prediction coverage)
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, (pad_left, pad_right), mode='constant', constant_values=0)

    # Bin the extracted window
    return bin_atac_track(
        window,
        source_resolution=source_resolution,
        target_bin_size=target_bin_size,
        aggregation=aggregation
    )


def extract_binned_features_borzoi(
    model,
    maternal_sequence: str,
    paternal_sequence: str,
    track_indices: List[int],
    tss: int,
    seq_center: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_bin_size: int = DEFAULT_BIN_SIZE,
    aggregation: Literal['mean', 'sum', 'max'] = 'mean',
    device: str = 'cuda',
    maternal_transformer: Optional['CoordinateTransformer'] = None,
    paternal_transformer: Optional['CoordinateTransformer'] = None,
    return_per_track: bool = False
) -> np.ndarray:
    """
    Extract binned ATAC features using Borzoi model.

    Args:
        model: Borzoi model (AnnotatedBorzoi or Flashzoi)
        maternal_sequence: Maternal haplotype sequence (524KB)
        paternal_sequence: Paternal haplotype sequence (524KB)
        track_indices: DNASE track indices for target tissue
        tss: TSS position (genomic coordinates)
        seq_center: Center of the sequence (genomic coordinates)
        window_size: Window around TSS (default: 100KB)
        target_bin_size: Target bin size (default: 256bp)
        aggregation: Aggregation method (default: 'mean')
        device: Device for inference
        maternal_transformer: Optional coordinate transformer for maternal
        paternal_transformer: Optional coordinate transformer for paternal
        return_per_track: If True, return features per track instead of averaging.
                         Shape: (n_tracks, n_bins) instead of (n_bins,)

    Returns:
        If return_per_track=False: Feature vector of shape (n_bins,)
        If return_per_track=True: Feature matrix of shape (n_tracks, n_bins)
        For 100KB/256bp: n_bins = 390

    Notes:
        - Borzoi output: 6144 bins at 32bp = 196KB coverage
        - Extracts central window around TSS
        - Averages across specified DNASE tracks (unless return_per_track=True)
        - Predicts on both haplotypes and averages
    """
    from .borzoi_utils import (
        predict_borzoi,
        BORZOI_BIN_SIZE, BORZOI_NUM_BINS, BORZOI_PRED_LENGTH
    )

    # predict_borzoi takes sequence string and returns (n_tracks, n_bins) numpy array
    # Get predictions for both haplotypes
    mat_preds = predict_borzoi(
        model, maternal_sequence,
        target_tracks=track_indices if track_indices else None,
        device=device
    )  # Shape: (len(track_indices), 6144) or (7611, 6144) if no tracks specified

    pat_preds = predict_borzoi(
        model, paternal_sequence,
        target_tracks=track_indices if track_indices else None,
        device=device
    )

    # Calculate TSS position in bin space
    # Borzoi predicts center PRED_LENGTH of input
    pred_half_length = BORZOI_PRED_LENGTH // 2  # 98,304 bp
    pred_start = seq_center - pred_half_length

    # Handle coordinate transformation for personal genomes
    if maternal_transformer is not None:
        mat_tss = maternal_transformer.ref_to_personal(tss)
    else:
        mat_tss = tss

    if paternal_transformer is not None:
        pat_tss = paternal_transformer.ref_to_personal(tss)
    else:
        pat_tss = tss

    # Convert TSS to bin index
    mat_center_bin = (mat_tss - pred_start) // BORZOI_BIN_SIZE
    pat_center_bin = (pat_tss - pred_start) // BORZOI_BIN_SIZE

    # Clamp to valid range
    mat_center_bin = int(max(0, min(BORZOI_NUM_BINS - 1, mat_center_bin)))
    pat_center_bin = int(max(0, min(BORZOI_NUM_BINS - 1, pat_center_bin)))

    if return_per_track:
        # Return features per track: (n_tracks, n_bins)
        n_tracks = mat_preds.shape[0]
        n_bins = window_size // target_bin_size

        all_features = np.zeros((n_tracks, n_bins), dtype=np.float32)

        for track_i in range(n_tracks):
            # Extract features for this track from both haplotypes
            mat_features = bin_atac_around_tss(
                mat_preds[track_i],  # (6144,)
                source_resolution=BORZOI_BIN_SIZE,
                source_center_idx=mat_center_bin,
                window_size=window_size,
                target_bin_size=target_bin_size,
                aggregation=aggregation
            )

            pat_features = bin_atac_around_tss(
                pat_preds[track_i],  # (6144,)
                source_resolution=BORZOI_BIN_SIZE,
                source_center_idx=pat_center_bin,
                window_size=window_size,
                target_bin_size=target_bin_size,
                aggregation=aggregation
            )

            # Average haplotypes for this track
            all_features[track_i] = (mat_features + pat_features) / 2

        return all_features  # (n_tracks, n_bins)

    else:
        # Default: Average across tracks to get single ATAC signal
        mat_track = mat_preds.mean(axis=0)  # (n_tracks, 6144) -> (6144,)
        pat_track = pat_preds.mean(axis=0)  # (6144,)

        # Extract and bin around TSS for each haplotype
        mat_features = bin_atac_around_tss(
            mat_track,
            source_resolution=BORZOI_BIN_SIZE,
            source_center_idx=mat_center_bin,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        pat_features = bin_atac_around_tss(
            pat_track,
            source_resolution=BORZOI_BIN_SIZE,
            source_center_idx=pat_center_bin,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        # Average haplotypes
        features = (mat_features + pat_features) / 2

        return features  # (n_bins,)


def extract_binned_features_alphagenome(
    model,
    maternal_sequence: str,
    paternal_sequence: str,
    ontology_terms: List[str],
    tss: int,
    seq_center: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_bin_size: int = DEFAULT_BIN_SIZE,
    aggregation: Literal['mean', 'sum', 'max'] = 'mean',
    sequence_length_bp: int = 102400,
    maternal_transformer: Optional['CoordinateTransformer'] = None,
    paternal_transformer: Optional['CoordinateTransformer'] = None,
    output_type: Literal['atac', 'dnase'] = 'dnase'
) -> np.ndarray:
    """
    Extract binned chromatin accessibility features using AlphaGenome model.

    Args:
        model: AlphaGenome DNA model
        maternal_sequence: Maternal haplotype sequence
        paternal_sequence: Paternal haplotype sequence
        ontology_terms: Tissue ontology terms (e.g., ['UBERON:0009834'])
        tss: TSS position (genomic coordinates)
        seq_center: Center of the sequence (genomic coordinates)
        window_size: Window around TSS (default: 100KB)
        target_bin_size: Target bin size (default: 256bp)
        aggregation: Aggregation method (default: 'mean')
        sequence_length_bp: AlphaGenome sequence length in bp
        maternal_transformer: Optional coordinate transformer
        paternal_transformer: Optional coordinate transformer
        output_type: Type of chromatin accessibility output ('atac' or 'dnase')
                    Note: DNASE has more brain/tissue tracks available

    Returns:
        Feature vector of shape (window_size // target_bin_size,)
        For 100KB/256bp: shape = (390,)

    Notes:
        - AlphaGenome outputs at 1bp resolution
        - Predicts on both haplotypes and averages
        - Does NOT apply log1p (applied after StandardScaler in training)
    """
    try:
        from alphagenome.models import dna_client
    except ImportError:
        raise ImportError("alphagenome package required for AlphaGenome predictions")

    # Helper to pad/trim sequence to expected length
    def prepare_sequence(seq: str, target_len: int) -> str:
        current_len = len(seq)
        if current_len < target_len:
            return seq.center(target_len, 'N')
        elif current_len > target_len:
            excess = current_len - target_len
            trim = excess // 2
            return seq[trim:trim + target_len]
        return seq

    # Prepare sequences to match expected length
    mat_seq = prepare_sequence(maternal_sequence, sequence_length_bp)
    pat_seq = prepare_sequence(paternal_sequence, sequence_length_bp)

    # Select output type based on parameter
    if output_type == 'dnase':
        requested_output = dna_client.OutputType.DNASE
    else:
        requested_output = dna_client.OutputType.ATAC

    # Predict chromatin accessibility for both haplotypes
    mat_output = model.predict_sequence(
        sequence=mat_seq,
        requested_outputs=[requested_output],
        ontology_terms=ontology_terms
    )

    pat_output = model.predict_sequence(
        sequence=pat_seq,
        requested_outputs=[requested_output],
        ontology_terms=ontology_terms
    )

    # Extract tracks based on output type
    if output_type == 'dnase':
        mat_values = mat_output.dnase.values
        pat_values = pat_output.dnase.values
    else:
        mat_values = mat_output.atac.values
        pat_values = pat_output.atac.values

    # Check if we got valid data
    if mat_values.size == 0 or pat_values.size == 0:
        raise ValueError(f"Empty {output_type.upper()} predictions - check ontology_terms: {ontology_terms}")

    # Average across tissues if multiple (axis 1)
    if mat_values.ndim > 1 and mat_values.shape[1] > 0:
        mat_track = mat_values.mean(axis=1)  # Average across tissues -> (n_positions,)
        pat_track = pat_values.mean(axis=1)
    elif mat_values.ndim == 1:
        mat_track = mat_values
        pat_track = pat_values
    else:
        raise ValueError(f"Unexpected {output_type.upper()} shape: {mat_values.shape}")

    # Calculate TSS position in track
    half_seq = sequence_length_bp // 2

    # Handle coordinate transformation
    if maternal_transformer is not None:
        mat_tss = maternal_transformer.ref_to_personal(tss)
    else:
        mat_tss = tss

    if paternal_transformer is not None:
        pat_tss = paternal_transformer.ref_to_personal(tss)
    else:
        pat_tss = tss

    # TSS index in track (relative to sequence center)
    mat_center_idx = half_seq + (mat_tss - seq_center)
    pat_center_idx = half_seq + (pat_tss - seq_center)

    # Clamp to valid range
    mat_center_idx = max(0, min(len(mat_track) - 1, mat_center_idx))
    pat_center_idx = max(0, min(len(pat_track) - 1, pat_center_idx))

    # Extract and bin around TSS for each haplotype
    mat_features = bin_atac_around_tss(
        mat_track,
        source_resolution=ALPHAGENOME_RESOLUTION,
        source_center_idx=int(mat_center_idx),
        window_size=window_size,
        target_bin_size=target_bin_size,
        aggregation=aggregation
    )

    pat_features = bin_atac_around_tss(
        pat_track,
        source_resolution=ALPHAGENOME_RESOLUTION,
        source_center_idx=int(pat_center_idx),
        window_size=window_size,
        target_bin_size=target_bin_size,
        aggregation=aggregation
    )

    # Average haplotypes
    features = (mat_features + pat_features) / 2

    return features


def create_feature_matrix_for_gene(
    model,
    genome_dataset,
    gene_row: Dict,
    sample_ids: List[str],
    model_type: Literal['borzoi', 'alphagenome'],
    config: Dict,
    device: str = 'cuda',
    return_per_track: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature matrix from binned ATAC predictions for a single gene.

    Args:
        model: AlphaGenome or Borzoi model
        genome_dataset: GenomeDataset for efficient sequence extraction
        gene_row: Gene metadata dict with 'chr', 'tss', 'ensg'
        sample_ids: List of sample IDs to process
        model_type: 'alphagenome' or 'borzoi'
        config: Configuration dictionary with:
            - ontology_terms: Tissue terms (AlphaGenome)
            - track_indices: Track indices (Borzoi)
            - window_size: Window around TSS (default: 100000)
            - target_bin_size: Target bin size (default: 256)
            - sequence_length_bp: AlphaGenome sequence length
        device: Device for inference
        return_per_track: If True (Borzoi only), return features per track.
                         Shape: (n_samples, n_tracks, n_bins) instead of (n_samples, n_bins)

    Returns:
        Tuple of:
            - X: Feature matrix
                - If return_per_track=False: (n_valid_samples, n_bins)
                - If return_per_track=True: (n_valid_samples, n_tracks, n_bins)
            - valid_sample_ids: Sample IDs with successful predictions
    """
    window_size = config.get('window_size', DEFAULT_WINDOW_SIZE)
    target_bin_size = config.get('target_bin_size', DEFAULT_BIN_SIZE)
    aggregation = config.get('aggregation', 'mean')

    n_bins = window_size // target_bin_size
    features_list = []
    valid_samples = []

    gene_id = gene_row.get('ensg', gene_row.get('gene_id', 'unknown'))
    tss = int(gene_row['tss'])

    # Find gene index in the dataset
    gene_metadata = genome_dataset.gene_metadata
    gene_mask = gene_metadata['ensg'] == gene_id
    if not gene_mask.any():
        logger.warning(f"Gene {gene_id} not found in genome_dataset")
        return np.array([]).reshape(0, n_bins), []

    gene_idx = gene_mask.idxmax()

    # Get sample indices in the dataset
    dataset_samples = genome_dataset.sample_list
    sample_to_dataset_idx = {s: i for i, s in enumerate(dataset_samples)}

    for sample_id in sample_ids:
        if sample_id not in sample_to_dataset_idx:
            logger.debug(f"Sample {sample_id} not in genome_dataset")
            continue

        sample_idx = sample_to_dataset_idx[sample_id]

        try:
            # Calculate the flat index for this gene-sample pair
            # GenomeDataset uses: idx = gene_idx * n_samples + sample_idx
            flat_idx = gene_idx * genome_dataset.n_samples + sample_idx

            # Get data from dataset
            # Returns: (seq_arr, expr_arr, transformers) or with indices if return_idx=True
            data = genome_dataset[flat_idx]

            # Parse based on return_idx setting
            if genome_dataset.return_idx:
                seq_data, expr_arr, transformers, _, _ = data
            else:
                seq_data, expr_arr, transformers = data

            # Check for one-hot encoding (not supported)
            if genome_dataset.onehot_encode:
                logger.warning(f"One-hot encoded sequences not supported, skipping {sample_id}")
                continue

            # Parse sequence data
            # seq_data format depends on std_output:
            # - std_output=True: np.array([mat_seq, pat_seq], dtype=object)
            # - std_output=False: np.stack([reference, personal]) where personal = [mat, pat]
            if genome_dataset.std_output:
                # Direct mat/pat array
                mat_seq = str(seq_data[0])
                pat_seq = str(seq_data[1])
            else:
                # seq_data has shape (2, 2): [reference, personal]
                # reference = [ref, ref], personal = [mat, pat]
                personal = seq_data[1]  # The personal sequences [mat, pat]
                mat_seq = str(personal[0])
                pat_seq = str(personal[1])

            # Extract coordinate transformers
            mat_transformer, pat_transformer = transformers

            # Extract binned features
            if model_type == 'borzoi':
                track_indices = config.get('track_indices', [])
                features = extract_binned_features_borzoi(
                    model=model,
                    maternal_sequence=mat_seq,
                    paternal_sequence=pat_seq,
                    track_indices=track_indices,
                    tss=tss,
                    seq_center=tss,  # TSS is the center of the sequence
                    window_size=window_size,
                    target_bin_size=target_bin_size,
                    aggregation=aggregation,
                    device=device,
                    maternal_transformer=mat_transformer,
                    paternal_transformer=pat_transformer,
                    return_per_track=return_per_track
                )
            else:  # alphagenome
                ontology_terms = config.get('ontology_terms', [])
                sequence_length_bp = config.get('sequence_length_bp', 102400)
                output_type = config.get('alphagenome_output_type', 'dnase')  # Default to DNASE (more brain tracks)
                features = extract_binned_features_alphagenome(
                    model=model,
                    maternal_sequence=mat_seq,
                    paternal_sequence=pat_seq,
                    ontology_terms=ontology_terms,
                    tss=tss,
                    seq_center=tss,
                    window_size=window_size,
                    target_bin_size=target_bin_size,
                    aggregation=aggregation,
                    sequence_length_bp=sequence_length_bp,
                    maternal_transformer=mat_transformer,
                    paternal_transformer=pat_transformer,
                    output_type=output_type
                )

            # Validate feature dimensions
            if return_per_track and model_type == 'borzoi':
                # features shape: (n_tracks, n_bins)
                expected_shape = (len(config.get('track_indices', [])), n_bins)
                if features.shape != expected_shape:
                    logger.warning(
                        f"Feature shape mismatch for {sample_id}, gene {gene_id}: "
                        f"expected {expected_shape}, got {features.shape}"
                    )
                    continue
            else:
                # features shape: (n_bins,)
                if len(features) != n_bins:
                    logger.warning(
                        f"Feature dimension mismatch for {sample_id}, gene {gene_id}: "
                        f"expected {n_bins}, got {len(features)}"
                    )
                    # Pad or truncate to expected size
                    if len(features) < n_bins:
                        features = np.pad(features, (0, n_bins - len(features)), mode='constant')
                    else:
                        features = features[:n_bins]

            features_list.append(features)
            valid_samples.append(sample_id)

        except Exception as e:
            logger.warning(f"Error extracting features for {sample_id}, gene {gene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(features_list) == 0:
        if return_per_track and model_type == 'borzoi':
            n_tracks = len(config.get('track_indices', []))
            return np.array([]).reshape(0, n_tracks, n_bins), []
        return np.array([]).reshape(0, n_bins), []

    X = np.stack(features_list, axis=0)
    # X shape: (n_samples, n_bins) or (n_samples, n_tracks, n_bins)
    return X, valid_samples


def save_all_predictions(
    output_path: Union[str, Path],
    predictions_dict: Dict[str, np.ndarray],
    gene_ids: List[str],
    sample_ids: List[str],
    track_info: Dict[str, Any],
    config: Dict[str, Any],
    gene_metadata: pd.DataFrame,
    model_type: Literal['borzoi', 'alphagenome'] = 'borzoi',
    compress: bool = True,
    valid_samples_per_gene: Optional[Dict[str, List[str]]] = None
) -> Path:
    """
    Save all binned predictions to a single combined NPZ file.

    This function combines per-gene prediction arrays into a single tensor
    for efficient storage and later analysis.

    Args:
        output_path: Destination path for the NPZ file.
        predictions_dict: Mapping from gene_id to prediction array.
            - Borzoi (per-track): shape (n_valid_samples, n_tracks, n_bins)
            - AlphaGenome/averaged: shape (n_valid_samples, n_bins)
        gene_ids: Ordered list of gene identifiers.
        sample_ids: Ordered list of all sample identifiers.
        track_info: Track metadata dictionary:
            - Borzoi: {'indices': List[int], 'descriptions': List[str]}
            - AlphaGenome: {'ontology_terms': List[str]}
        config: Configuration used for predictions:
            - 'window_size': int (e.g., 100000)
            - 'bin_size' or 'target_bin_size': int (e.g., 256)
            - 'aggregation': str (e.g., 'mean')
        gene_metadata: DataFrame with 'ensg'/'gene_id', 'chr', 'tss' columns.
        model_type: Type of model ('borzoi' or 'alphagenome').
        compress: If True, use np.savez_compressed (default: True).
        valid_samples_per_gene: Optional mapping from gene_id to list of
            valid sample IDs. Used to construct validity mask.

    Returns:
        Path to the saved NPZ file.

    Example:
        >>> save_all_predictions(
        ...     output_path='results/all_predictions.npz',
        ...     predictions_dict={'ENSG00001': X1, 'ENSG00002': X2},
        ...     gene_ids=['ENSG00001', 'ENSG00002'],
        ...     sample_ids=['sample1', 'sample2', 'sample3'],
        ...     track_info={'indices': [1288, 1336], 'descriptions': ['brain1', 'brain2']},
        ...     config={'window_size': 100000, 'bin_size': 256},
        ...     gene_metadata=gene_df,
        ...     model_type='borzoi'
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_genes = len(gene_ids)
    n_samples = len(sample_ids)

    # Determine dimensions from first prediction
    first_gene = gene_ids[0]
    first_pred = predictions_dict[first_gene]

    # Determine if per-track mode based on array dimensions
    if first_pred.ndim == 3:
        # Per-track mode: (n_valid_samples, n_tracks, n_bins)
        n_tracks = first_pred.shape[1]
        n_bins = first_pred.shape[2]
        per_track_mode = True
    else:
        # Averaged mode: (n_valid_samples, n_bins)
        n_tracks = 1
        n_bins = first_pred.shape[1]
        per_track_mode = False

    # Create sample index mapping
    sample_to_idx = {s: i for i, s in enumerate(sample_ids)}

    # Initialize arrays
    if per_track_mode:
        predictions = np.zeros((n_genes, n_samples, n_tracks, n_bins), dtype=np.float32)
    else:
        predictions = np.zeros((n_genes, n_samples, n_bins), dtype=np.float32)

    valid_mask = np.zeros((n_genes, n_samples), dtype=bool)

    # Fill in predictions
    for gene_idx, gene_id in enumerate(gene_ids):
        if gene_id not in predictions_dict:
            logger.warning(f"Gene {gene_id} not in predictions_dict, skipping")
            continue

        pred_array = predictions_dict[gene_id]

        # Get valid samples for this gene
        if valid_samples_per_gene and gene_id in valid_samples_per_gene:
            gene_valid_samples = valid_samples_per_gene[gene_id]
        else:
            # Assume all samples are valid if not specified
            gene_valid_samples = sample_ids[:pred_array.shape[0]]

        # Map predictions to global sample indices
        for local_idx, sample_id in enumerate(gene_valid_samples):
            if sample_id in sample_to_idx and local_idx < pred_array.shape[0]:
                global_idx = sample_to_idx[sample_id]
                predictions[gene_idx, global_idx] = pred_array[local_idx]
                valid_mask[gene_idx, global_idx] = True

    # Prepare gene metadata arrays
    gene_chr = []
    gene_tss = []
    gene_id_col = 'ensg' if 'ensg' in gene_metadata.columns else 'gene_id'

    for gene_id in gene_ids:
        gene_row = gene_metadata[gene_metadata[gene_id_col] == gene_id]
        if len(gene_row) > 0:
            gene_chr.append(str(gene_row['chr'].iloc[0]))
            gene_tss.append(int(gene_row['tss'].iloc[0]))
        else:
            gene_chr.append('')
            gene_tss.append(0)

    # Prepare track info arrays
    if model_type == 'borzoi' and 'indices' in track_info:
        track_ids = np.array(track_info.get('indices', []), dtype=np.int64)
        track_descriptions = np.array(track_info.get('descriptions', []), dtype=object)
    else:
        # AlphaGenome or no track info
        track_ids = np.array(track_info.get('ontology_terms', []), dtype=object)
        track_descriptions = np.array(track_info.get('ontology_terms', []), dtype=object)

    # Build save dict
    save_dict = {
        # Core predictions
        'predictions': predictions,
        'valid_mask': valid_mask,

        # Index arrays
        'gene_ids': np.array(gene_ids, dtype=object),
        'sample_ids': np.array(sample_ids, dtype=object),
        'track_ids': track_ids,
        'track_descriptions': track_descriptions,

        # Gene metadata
        'gene_chr': np.array(gene_chr, dtype=object),
        'gene_tss': np.array(gene_tss, dtype=np.int64),

        # Configuration
        'window_size': np.int64(config.get('window_size', DEFAULT_WINDOW_SIZE)),
        'bin_size': np.int64(config.get('bin_size', config.get('target_bin_size', DEFAULT_BIN_SIZE))),
        'n_bins': np.int64(n_bins),
        'n_tracks': np.int64(n_tracks),
        'n_genes': np.int64(n_genes),
        'n_samples': np.int64(n_samples),
        'model_type': np.array(model_type, dtype=object),
        'aggregation': np.array(config.get('aggregation', 'mean'), dtype=object),
        'per_track_mode': np.array(per_track_mode, dtype=bool),

        # Timestamp
        'created_at': np.array(datetime.now().isoformat(), dtype=object),
    }

    # Save
    if compress:
        np.savez_compressed(output_path, **save_dict)
    else:
        np.savez(output_path, **save_dict)

    logger.info(
        f"Saved all predictions to {output_path}: "
        f"{n_genes} genes, {n_samples} samples, {n_tracks} tracks, {n_bins} bins"
    )
    logger.info(f"Valid predictions: {valid_mask.sum()} / {n_genes * n_samples} ({100*valid_mask.mean():.1f}%)")

    return output_path


def load_all_predictions(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load combined predictions from NPZ file.

    Args:
        file_path: Path to the NPZ file saved by save_all_predictions().

    Returns:
        Dictionary containing:
            - 'predictions': np.ndarray - Full prediction tensor
            - 'gene_ids': np.ndarray - Gene identifiers
            - 'sample_ids': np.ndarray - Sample identifiers
            - 'track_ids': np.ndarray - Track indices or ontology terms
            - 'track_descriptions': np.ndarray - Track descriptions
            - 'valid_mask': np.ndarray - Boolean validity mask
            - 'gene_chr': np.ndarray - Chromosome per gene
            - 'gene_tss': np.ndarray - TSS per gene
            - 'config': Dict - Configuration used (window_size, bin_size, etc.)
            - 'model_type': str - 'borzoi' or 'alphagenome'
            - 'per_track_mode': bool - Whether per-track mode was used
            - 'created_at': str - ISO timestamp

    Example:
        >>> data = load_all_predictions('results/all_predictions.npz')
        >>> predictions = data['predictions']  # (n_genes, n_samples, [n_tracks], n_bins)
        >>> gene_ids = data['gene_ids']
        >>> # Access predictions for specific gene and sample
        >>> gene_idx = np.where(gene_ids == 'ENSG00000001')[0][0]
        >>> sample_idx = np.where(data['sample_ids'] == 'sample1')[0][0]
        >>> if data['valid_mask'][gene_idx, sample_idx]:
        ...     pred = predictions[gene_idx, sample_idx]
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)

    # Extract configuration
    config = {
        'window_size': int(data['window_size']),
        'bin_size': int(data['bin_size']),
        'n_bins': int(data['n_bins']),
        'n_tracks': int(data['n_tracks']),
        'n_genes': int(data['n_genes']),
        'n_samples': int(data['n_samples']),
        'aggregation': str(data['aggregation']),
    }

    result = {
        # Core data
        'predictions': data['predictions'],
        'valid_mask': data['valid_mask'],

        # Index arrays
        'gene_ids': data['gene_ids'],
        'sample_ids': data['sample_ids'],
        'track_ids': data['track_ids'],
        'track_descriptions': data['track_descriptions'],

        # Gene metadata
        'gene_chr': data['gene_chr'],
        'gene_tss': data['gene_tss'],

        # Configuration
        'config': config,
        'model_type': str(data['model_type']),
        'per_track_mode': bool(data['per_track_mode']),
        'created_at': str(data['created_at']),
    }

    logger.info(
        f"Loaded predictions from {file_path}: "
        f"{config['n_genes']} genes, {config['n_samples']} samples, "
        f"{config['n_tracks']} tracks, {config['n_bins']} bins"
    )

    return result


# =============================================================================
# RNA-specific binning functions
# =============================================================================


def load_rna_track_pairs_from_csv(csv_path: str) -> Tuple[List[Tuple[int, int]], pd.DataFrame]:
    """
    Load RNA track pairs from CSV file.

    RNA tracks come in paired +/- strand configurations. This function parses
    the track CSV and returns pairs of (plus_idx, minus_idx) for each sample.

    Args:
        csv_path: Path to RNA tracks CSV file with columns:
            - index: Track index in Borzoi output
            - identifier: ENCODE file accession (e.g., 'ENCFF196HWN+')
            - strand_pair: Index of paired strand track
            - description: Track description
            - strand: '+' or '-'

    Returns:
        Tuple of:
            - track_pairs: List of (plus_idx, minus_idx) tuples
            - tracks_df: Full DataFrame with track metadata

    Example:
        >>> pairs, df = load_rna_track_pairs_from_csv('selected_tracks_rosmap_rna.csv')
        >>> print(f"Found {len(pairs)} track pairs")
        >>> for plus_idx, minus_idx in pairs[:3]:
        ...     print(f"  +strand: {plus_idx}, -strand: {minus_idx}")
    """
    tracks_df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['index', 'strand', 'strand_pair']
    for col in required_cols:
        if col not in tracks_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract track pairs by finding + strand tracks and their - strand pairs
    plus_tracks = tracks_df[tracks_df['strand'] == '+']

    track_pairs = []
    for _, row in plus_tracks.iterrows():
        plus_idx = int(row['index'])
        minus_idx = int(row['strand_pair'])
        track_pairs.append((plus_idx, minus_idx))

    logger.info(f"Loaded {len(track_pairs)} RNA track pairs from {csv_path}")

    return track_pairs, tracks_df


def extract_binned_features_borzoi_rna(
    model,
    maternal_sequence: str,
    paternal_sequence: str,
    track_pairs: List[Tuple[int, int]],
    tss: int,
    seq_center: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_bin_size: int = DEFAULT_BIN_SIZE,
    aggregation: Literal['mean', 'sum', 'max'] = 'mean',
    device: str = 'cuda',
    maternal_transformer: Optional['CoordinateTransformer'] = None,
    paternal_transformer: Optional['CoordinateTransformer'] = None,
    return_per_track: bool = False
) -> np.ndarray:
    """
    Extract binned RNA features using Borzoi model with paired strand tracks.

    RNA tracks in Borzoi come in +/- strand pairs. This function:
    1. Runs single inference per haplotype to get all tracks
    2. Sums +/- strand predictions bin-wise per track pair
    3. Averages maternal/paternal haplotypes
    4. Bins around TSS to target resolution

    Args:
        model: Borzoi model (AnnotatedBorzoi or Flashzoi)
        maternal_sequence: Maternal haplotype sequence (524KB)
        paternal_sequence: Paternal haplotype sequence (524KB)
        track_pairs: List of (plus_idx, minus_idx) tuples for each RNA sample
        tss: TSS position (genomic coordinates)
        seq_center: Center of the sequence (genomic coordinates)
        window_size: Window around TSS (default: 100KB)
        target_bin_size: Target bin size (default: 256bp)
        aggregation: Aggregation method (default: 'mean')
        device: Device for inference
        maternal_transformer: Optional coordinate transformer for maternal
        paternal_transformer: Optional coordinate transformer for paternal
        return_per_track: If True, return features per track pair instead of averaging.
                         Shape: (n_track_pairs, n_bins) instead of (n_bins,)

    Returns:
        If return_per_track=False: Feature vector of shape (n_bins,)
        If return_per_track=True: Feature matrix of shape (n_track_pairs, n_bins)
        For 100KB/256bp: n_bins = 390

    Notes:
        - Runs ONE inference per haplotype (efficient for multiple tracks)
        - Each track pair is summed bin-wise (+/- strand)
        - Haplotypes are averaged
    """
    from .borzoi_utils import (
        predict_borzoi,
        BORZOI_BIN_SIZE, BORZOI_NUM_BINS, BORZOI_PRED_LENGTH
    )

    # Collect all unique track indices for single inference
    all_track_indices = []
    for plus_idx, minus_idx in track_pairs:
        if plus_idx not in all_track_indices:
            all_track_indices.append(plus_idx)
        if minus_idx not in all_track_indices:
            all_track_indices.append(minus_idx)

    # Create index mapping for efficient lookup
    track_to_local_idx = {idx: i for i, idx in enumerate(all_track_indices)}

    # Run single inference for all tracks per haplotype
    mat_preds = predict_borzoi(
        model, maternal_sequence,
        target_tracks=all_track_indices,
        device=device
    )  # Shape: (len(all_track_indices), 6144)

    pat_preds = predict_borzoi(
        model, paternal_sequence,
        target_tracks=all_track_indices,
        device=device
    )

    # Calculate TSS position in bin space
    pred_half_length = BORZOI_PRED_LENGTH // 2  # 98,304 bp
    pred_start = seq_center - pred_half_length

    # Handle coordinate transformation for personal genomes
    if maternal_transformer is not None:
        mat_tss = maternal_transformer.ref_to_personal(tss)
    else:
        mat_tss = tss

    if paternal_transformer is not None:
        pat_tss = paternal_transformer.ref_to_personal(tss)
    else:
        pat_tss = tss

    # Convert TSS to bin index
    mat_center_bin = (mat_tss - pred_start) // BORZOI_BIN_SIZE
    pat_center_bin = (pat_tss - pred_start) // BORZOI_BIN_SIZE

    # Clamp to valid range
    mat_center_bin = int(max(0, min(BORZOI_NUM_BINS - 1, mat_center_bin)))
    pat_center_bin = int(max(0, min(BORZOI_NUM_BINS - 1, pat_center_bin)))

    n_bins = window_size // target_bin_size
    n_track_pairs = len(track_pairs)

    if return_per_track:
        # Return features per track pair: (n_track_pairs, n_bins)
        all_features = np.zeros((n_track_pairs, n_bins), dtype=np.float32)

        for pair_i, (plus_idx, minus_idx) in enumerate(track_pairs):
            plus_local = track_to_local_idx[plus_idx]
            minus_local = track_to_local_idx[minus_idx]

            # Sum +/- strands bin-wise for each haplotype
            mat_combined = mat_preds[plus_local] + mat_preds[minus_local]  # (6144,)
            pat_combined = pat_preds[plus_local] + pat_preds[minus_local]  # (6144,)

            # Extract and bin around TSS
            mat_features = bin_atac_around_tss(
                mat_combined,
                source_resolution=BORZOI_BIN_SIZE,
                source_center_idx=mat_center_bin,
                window_size=window_size,
                target_bin_size=target_bin_size,
                aggregation=aggregation
            )

            pat_features = bin_atac_around_tss(
                pat_combined,
                source_resolution=BORZOI_BIN_SIZE,
                source_center_idx=pat_center_bin,
                window_size=window_size,
                target_bin_size=target_bin_size,
                aggregation=aggregation
            )

            # Average haplotypes
            all_features[pair_i] = (mat_features + pat_features) / 2

        return all_features  # (n_track_pairs, n_bins)

    else:
        # Average across all track pairs
        combined_features = np.zeros(n_bins, dtype=np.float32)

        for plus_idx, minus_idx in track_pairs:
            plus_local = track_to_local_idx[plus_idx]
            minus_local = track_to_local_idx[minus_idx]

            # Sum +/- strands bin-wise for each haplotype
            mat_combined = mat_preds[plus_local] + mat_preds[minus_local]
            pat_combined = pat_preds[plus_local] + pat_preds[minus_local]

            # Extract and bin around TSS
            mat_features = bin_atac_around_tss(
                mat_combined,
                source_resolution=BORZOI_BIN_SIZE,
                source_center_idx=mat_center_bin,
                window_size=window_size,
                target_bin_size=target_bin_size,
                aggregation=aggregation
            )

            pat_features = bin_atac_around_tss(
                pat_combined,
                source_resolution=BORZOI_BIN_SIZE,
                source_center_idx=pat_center_bin,
                window_size=window_size,
                target_bin_size=target_bin_size,
                aggregation=aggregation
            )

            # Average haplotypes and accumulate
            combined_features += (mat_features + pat_features) / 2

        # Average across track pairs
        combined_features /= n_track_pairs

        return combined_features  # (n_bins,)


def extract_binned_features_alphagenome_rna(
    model,
    maternal_sequence: str,
    paternal_sequence: str,
    tss: int,
    seq_center: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_bin_size: int = DEFAULT_BIN_SIZE,
    aggregation: Literal['mean', 'sum', 'max'] = 'mean',
    sequence_length_bp: int = 102400,
    maternal_transformer: Optional['CoordinateTransformer'] = None,
    paternal_transformer: Optional['CoordinateTransformer'] = None,
    return_per_track: bool = False
) -> np.ndarray:
    """
    Extract binned RNA features using AlphaGenome model.

    AlphaGenome RNA output has 3 fixed tracks: [encode+, encode-, gtex]
    This function:
    1. Gets RNA predictions for both haplotypes
    2. Combines encode+ and encode- into encode_combined
    3. Returns either averaged features or per-track features

    Args:
        model: AlphaGenome DNA model
        maternal_sequence: Maternal haplotype sequence
        paternal_sequence: Paternal haplotype sequence
        tss: TSS position (genomic coordinates)
        seq_center: Center of the sequence (genomic coordinates)
        window_size: Window around TSS (default: 100KB)
        target_bin_size: Target bin size (default: 256bp)
        aggregation: Aggregation method (default: 'mean')
        sequence_length_bp: AlphaGenome sequence length in bp
        maternal_transformer: Optional coordinate transformer
        paternal_transformer: Optional coordinate transformer
        return_per_track: If True, return features for encode_combined and gtex separately.
                         Shape: (2, n_bins) instead of (n_bins,)

    Returns:
        If return_per_track=False: Feature vector of shape (n_bins,)
        If return_per_track=True: Feature matrix of shape (2, n_bins)
            - Track 0: encode_combined (sum of encode+ and encode-)
            - Track 1: gtex
        For 100KB/256bp: n_bins = 390

    Notes:
        - AlphaGenome RNA output: rna_seq.values shape (sequence_length, 3)
        - Track indices: [encode+ (0), encode- (1), gtex (2)]
    """
    try:
        from alphagenome.models import dna_client
    except ImportError:
        raise ImportError("alphagenome package required for AlphaGenome predictions")

    # Helper to pad/trim sequence to expected length
    def prepare_sequence(seq: str, target_len: int) -> str:
        current_len = len(seq)
        if current_len < target_len:
            return seq.center(target_len, 'N')
        elif current_len > target_len:
            excess = current_len - target_len
            trim = excess // 2
            return seq[trim:trim + target_len]
        return seq

    # Prepare sequences
    mat_seq = prepare_sequence(maternal_sequence, sequence_length_bp)
    pat_seq = prepare_sequence(paternal_sequence, sequence_length_bp)

    # Predict RNA for both haplotypes
    mat_output = model.predict_sequence(
        sequence=mat_seq,
        requested_outputs=[dna_client.OutputType.RNA_SEQ],
        ontology_terms=[]  # RNA output doesn't require ontology terms
    )

    pat_output = model.predict_sequence(
        sequence=pat_seq,
        requested_outputs=[dna_client.OutputType.RNA_SEQ],
        ontology_terms=[]
    )

    # Extract RNA values: shape (sequence_length, 3)
    # Tracks: [encode+ (0), encode- (1), gtex (2)]
    mat_values = mat_output.rna_seq.values
    pat_values = pat_output.rna_seq.values

    if mat_values.size == 0 or pat_values.size == 0:
        raise ValueError("Empty RNA predictions from AlphaGenome")

    # Calculate TSS position in track
    half_seq = sequence_length_bp // 2

    # Handle coordinate transformation
    if maternal_transformer is not None:
        mat_tss = maternal_transformer.ref_to_personal(tss)
    else:
        mat_tss = tss

    if paternal_transformer is not None:
        pat_tss = paternal_transformer.ref_to_personal(tss)
    else:
        pat_tss = tss

    # TSS index in track (relative to sequence center)
    mat_center_idx = half_seq + (mat_tss - seq_center)
    pat_center_idx = half_seq + (pat_tss - seq_center)

    # Clamp to valid range
    mat_center_idx = int(max(0, min(len(mat_values) - 1, mat_center_idx)))
    pat_center_idx = int(max(0, min(len(pat_values) - 1, pat_center_idx)))

    n_bins = window_size // target_bin_size

    if return_per_track:
        # Return 2 tracks: encode_combined and gtex
        all_features = np.zeros((2, n_bins), dtype=np.float32)

        # Track 0: encode_combined (sum of encode+ and encode-)
        mat_encode_combined = mat_values[:, 0] + mat_values[:, 1]
        pat_encode_combined = pat_values[:, 0] + pat_values[:, 1]

        mat_encode_features = bin_atac_around_tss(
            mat_encode_combined,
            source_resolution=ALPHAGENOME_RESOLUTION,
            source_center_idx=mat_center_idx,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        pat_encode_features = bin_atac_around_tss(
            pat_encode_combined,
            source_resolution=ALPHAGENOME_RESOLUTION,
            source_center_idx=pat_center_idx,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        all_features[0] = (mat_encode_features + pat_encode_features) / 2

        # Track 1: gtex
        mat_gtex = mat_values[:, 2]
        pat_gtex = pat_values[:, 2]

        mat_gtex_features = bin_atac_around_tss(
            mat_gtex,
            source_resolution=ALPHAGENOME_RESOLUTION,
            source_center_idx=mat_center_idx,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        pat_gtex_features = bin_atac_around_tss(
            pat_gtex,
            source_resolution=ALPHAGENOME_RESOLUTION,
            source_center_idx=pat_center_idx,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        all_features[1] = (mat_gtex_features + pat_gtex_features) / 2

        return all_features  # (2, n_bins)

    else:
        # Average all tracks: encode_combined and gtex
        # Sum encode+ and encode- first, then average with gtex
        mat_encode_combined = mat_values[:, 0] + mat_values[:, 1]
        pat_encode_combined = pat_values[:, 0] + pat_values[:, 1]

        mat_gtex = mat_values[:, 2]
        pat_gtex = pat_values[:, 2]

        # Average encode_combined and gtex
        mat_avg = (mat_encode_combined + mat_gtex) / 2
        pat_avg = (pat_encode_combined + pat_gtex) / 2

        mat_features = bin_atac_around_tss(
            mat_avg,
            source_resolution=ALPHAGENOME_RESOLUTION,
            source_center_idx=mat_center_idx,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        pat_features = bin_atac_around_tss(
            pat_avg,
            source_resolution=ALPHAGENOME_RESOLUTION,
            source_center_idx=pat_center_idx,
            window_size=window_size,
            target_bin_size=target_bin_size,
            aggregation=aggregation
        )

        features = (mat_features + pat_features) / 2

        return features  # (n_bins,)


def create_feature_matrix_for_gene_rna(
    model,
    genome_dataset,
    gene_row: Dict,
    sample_ids: List[str],
    model_type: Literal['borzoi', 'alphagenome'],
    config: Dict,
    device: str = 'cuda',
    return_per_track: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature matrix from binned RNA predictions for a single gene.

    This function wraps the RNA-specific feature extraction functions and
    handles sample iteration and error handling.

    Args:
        model: AlphaGenome or Borzoi model
        genome_dataset: GenomeDataset for efficient sequence extraction
        gene_row: Gene metadata dict with 'chr', 'tss', 'ensg'
        sample_ids: List of sample IDs to process
        model_type: 'alphagenome' or 'borzoi'
        config: Configuration dictionary with:
            - track_pairs: List of (plus_idx, minus_idx) tuples (Borzoi)
            - window_size: Window around TSS (default: 100000)
            - target_bin_size: Target bin size (default: 256)
            - sequence_length_bp: AlphaGenome sequence length
        device: Device for inference
        return_per_track: If True, return features per track.
            - Borzoi: (n_samples, n_track_pairs, n_bins)
            - AlphaGenome: (n_samples, 2, n_bins) for encode_combined and gtex

    Returns:
        Tuple of:
            - X: Feature matrix
                - If return_per_track=False: (n_valid_samples, n_bins)
                - If return_per_track=True: (n_valid_samples, n_tracks, n_bins)
            - valid_sample_ids: Sample IDs with successful predictions
    """
    window_size = config.get('window_size', DEFAULT_WINDOW_SIZE)
    target_bin_size = config.get('target_bin_size', DEFAULT_BIN_SIZE)
    aggregation = config.get('aggregation', 'mean')

    n_bins = window_size // target_bin_size
    features_list = []
    valid_samples = []

    gene_id = gene_row.get('ensg', gene_row.get('gene_id', 'unknown'))
    tss = int(gene_row['tss'])

    # Find gene index in the dataset
    gene_metadata = genome_dataset.gene_metadata
    gene_mask = gene_metadata['ensg'] == gene_id
    if not gene_mask.any():
        logger.warning(f"Gene {gene_id} not found in genome_dataset")
        if return_per_track:
            if model_type == 'borzoi':
                n_tracks = len(config.get('track_pairs', []))
            else:
                n_tracks = 2  # encode_combined, gtex
            return np.array([]).reshape(0, n_tracks, n_bins), []
        return np.array([]).reshape(0, n_bins), []

    gene_idx = gene_mask.idxmax()

    # Get sample indices in the dataset
    dataset_samples = genome_dataset.sample_list
    sample_to_dataset_idx = {s: i for i, s in enumerate(dataset_samples)}

    for sample_id in sample_ids:
        if sample_id not in sample_to_dataset_idx:
            logger.debug(f"Sample {sample_id} not in genome_dataset")
            continue

        sample_idx = sample_to_dataset_idx[sample_id]

        try:
            # Calculate the flat index for this gene-sample pair
            flat_idx = gene_idx * genome_dataset.n_samples + sample_idx

            # Get data from dataset
            data = genome_dataset[flat_idx]

            # Parse based on return_idx setting
            if genome_dataset.return_idx:
                seq_data, expr_arr, transformers, _, _ = data
            else:
                seq_data, expr_arr, transformers = data

            # Check for one-hot encoding (not supported)
            if genome_dataset.onehot_encode:
                logger.warning(f"One-hot encoded sequences not supported, skipping {sample_id}")
                continue

            # Parse sequence data
            if genome_dataset.std_output:
                mat_seq = str(seq_data[0])
                pat_seq = str(seq_data[1])
            else:
                personal = seq_data[1]
                mat_seq = str(personal[0])
                pat_seq = str(personal[1])

            # Extract coordinate transformers
            mat_transformer, pat_transformer = transformers

            # Extract binned features
            if model_type == 'borzoi':
                track_pairs = config.get('track_pairs', [])
                features = extract_binned_features_borzoi_rna(
                    model=model,
                    maternal_sequence=mat_seq,
                    paternal_sequence=pat_seq,
                    track_pairs=track_pairs,
                    tss=tss,
                    seq_center=tss,
                    window_size=window_size,
                    target_bin_size=target_bin_size,
                    aggregation=aggregation,
                    device=device,
                    maternal_transformer=mat_transformer,
                    paternal_transformer=pat_transformer,
                    return_per_track=return_per_track
                )
            else:  # alphagenome
                sequence_length_bp = config.get('sequence_length_bp', 102400)
                features = extract_binned_features_alphagenome_rna(
                    model=model,
                    maternal_sequence=mat_seq,
                    paternal_sequence=pat_seq,
                    tss=tss,
                    seq_center=tss,
                    window_size=window_size,
                    target_bin_size=target_bin_size,
                    aggregation=aggregation,
                    sequence_length_bp=sequence_length_bp,
                    maternal_transformer=mat_transformer,
                    paternal_transformer=pat_transformer,
                    return_per_track=return_per_track
                )

            features_list.append(features)
            valid_samples.append(sample_id)

        except Exception as e:
            logger.warning(f"Error extracting RNA features for {sample_id}, gene {gene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(features_list) == 0:
        if return_per_track:
            if model_type == 'borzoi':
                n_tracks = len(config.get('track_pairs', []))
            else:
                n_tracks = 2
            return np.array([]).reshape(0, n_tracks, n_bins), []
        return np.array([]).reshape(0, n_bins), []

    X = np.stack(features_list, axis=0)
    return X, valid_samples


__all__ = [
    # Constants
    'ALPHAGENOME_RESOLUTION',
    'BORZOI_RESOLUTION',
    'DEFAULT_BIN_SIZE',
    'DEFAULT_WINDOW_SIZE',
    # Core binning functions
    'bin_atac_track',
    'bin_atac_around_tss',
    # ATAC feature extraction
    'extract_binned_features_borzoi',
    'extract_binned_features_alphagenome',
    'create_feature_matrix_for_gene',
    # RNA feature extraction
    'load_rna_track_pairs_from_csv',
    'extract_binned_features_borzoi_rna',
    'extract_binned_features_alphagenome_rna',
    'create_feature_matrix_for_gene_rna',
    # Prediction saving/loading
    'save_all_predictions',
    'load_all_predictions',
]
