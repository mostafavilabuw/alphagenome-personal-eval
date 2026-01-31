"""
AlphaGenome prediction utilities.

This module provides simple wrapper functions for AlphaGenome predictions,
handling both ATAC-seq and RNA-seq predictions with personal genome sequences.
"""

import numpy as np
from typing import List, Optional, Literal, Tuple, Union, Dict, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from alphagenome_eval.utils.coordinates import CoordinateTransformer


# Sequence length constants (matching AlphaGenome API constants)
SEQUENCE_LENGTHS = {
    '2KB': 2048,
    '16KB': 16384,
    '100KB': 102400,
    '500KB': 524288,
    '1MB': 1048576
}


def init_dna_model(
    api_key: str,
    sequence_length: str = '100KB'
) -> Tuple[object, int]:
    """
    Initialize AlphaGenome DNA model.

    Args:
        api_key: AlphaGenome API key
        sequence_length: Sequence length ('2KB', '16KB', '100KB', '500KB', '1MB')

    Returns:
        model: AlphaGenome DNA model object
        sequence_length_bp: Sequence length in base pairs

    Example:
        >>> model, seq_len = init_dna_model(API_KEY, sequence_length='100KB')
        >>> print(f"Model initialized with {seq_len} bp sequences")
    """
    try:
        from alphagenome.models import dna_client
    except ImportError:
        raise ImportError(
            "AlphaGenome package not installed. "
            "Please install with: pip install alphagenome"
        )

    # Initialize model
    model = dna_client.create(api_key)

    # Get sequence length constant
    seq_len_map = {
        '2KB': dna_client.SEQUENCE_LENGTH_2KB,
        '16KB': dna_client.SEQUENCE_LENGTH_16KB,
        '100KB': dna_client.SEQUENCE_LENGTH_100KB,
        '500KB': dna_client.SEQUENCE_LENGTH_500KB,
        '1MB': dna_client.SEQUENCE_LENGTH_1MB
    }

    if sequence_length not in seq_len_map:
        raise ValueError(
            f"Invalid sequence_length: {sequence_length}. "
            f"Must be one of {list(seq_len_map.keys())}"
        )

    sequence_length_bp = seq_len_map[sequence_length]

    return model, sequence_length_bp


def predict_atac(
    model,
    dna_sequence: str,
    ontology_terms: List[str],
    peak_start: int,
    peak_end: int,
    sequence_length_bp: Optional[int] = None,
    aggregate: Optional[str] = 'sum',
    center_pad: bool = True,
    coord_transformer: Optional["CoordinateTransformer"] = None
) -> Union[float, np.ndarray]:
    """
    Predict ATAC-seq signal from DNA sequence, aggregating over peak region only.

    Args:
        model: AlphaGenome DNA model
        dna_sequence: DNA sequence string
        ontology_terms: List of ontology term IDs (e.g., ['EFO:0010843'])
        peak_start: Start coordinate of peak region (genomic coordinates)
        peak_end: End coordinate of peak region (genomic coordinates)
        sequence_length_bp: Expected sequence length (None = auto-detect)
        aggregate: 'sum', 'mean', or None (return full track)
        center_pad: Center and pad sequence with 'N's

    Returns:
        Aggregated value (float) if aggregate is not None, else array for peak region only

    Note:
        The sequence should be centered on the peak center: (peak_start + peak_end) / 2
        Aggregation is performed ONLY over the peak region, not the entire sequence window.

    Example:
        >>> # Aggregate over peak region (1.5KB peak within 100KB sequence)
        >>> pred = predict_atac(
        ...     model, sequence, ['EFO:0010843'],
        ...     peak_start=1045000, peak_end=1046500
        ... )
        >>> print(f"ATAC signal in peak: {pred}")
    """
    try:
        from alphagenome.models import dna_client
    except ImportError:
        raise ImportError("AlphaGenome package not installed")

    # Center and pad/truncate sequence if requested
    if center_pad and sequence_length_bp is not None:
        current_length = len(dna_sequence)
        
        if current_length < sequence_length_bp:
            # Pad with N's to center the sequence
            dna_sequence = dna_sequence.center(sequence_length_bp, 'N')
        elif current_length > sequence_length_bp:
            # Truncate from both ends to center the sequence
            excess = current_length - sequence_length_bp
            trim_left = excess // 2
            trim_right = excess - trim_left
            dna_sequence = dna_sequence[trim_left:current_length - trim_right]
        # else: sequence is already the correct length, no changes needed

    try:
        # Make prediction
        output = model.predict_sequence(
            sequence=dna_sequence,
            requested_outputs=[dna_client.OutputType.ATAC],
            ontology_terms=ontology_terms
        )

        # Extract ATAC values
        atac_values = output.atac.values  # Shape: (sequence_length, n_tracks)

        # Transform coordinates if transformer provided (for personalized genomes)
        if coord_transformer is not None:
            peak_start, peak_end = coord_transformer.transform_interval(peak_start, peak_end)

        # Calculate peak region within sequence
        # Sequence is centered on peak center
        seq_center_idx = atac_values.shape[0] // 2
        peak_center = (peak_start + peak_end) // 2
        peak_length = peak_end - peak_start
        # Calculate peak boundaries relative to sequence center
        # peak_start is (peak_center - peak_length/2) away from center
        # peak_end is (peak_center + peak_length/2) away from center
        start_idx = seq_center_idx - (peak_center - peak_start)
        end_idx = seq_center_idx + (peak_end - peak_center)
        
        # Clip to valid range
        start_idx = max(0, start_idx)
        end_idx = min(atac_values.shape[0], end_idx)
        
        # Extract peak region only
        peak_region_values = atac_values[start_idx:end_idx, :]

        # Aggregate if requested
        if aggregate == 'sum':
            agg_value = float(np.sum(peak_region_values))
            # Apply log1p to match log-transformed observed FPKM values
            return np.log1p(agg_value)
        elif aggregate == 'mean':
            agg_value = float(np.mean(peak_region_values))
            # Apply log1p to match log-transformed observed FPKM values
            return np.log1p(agg_value)
        elif aggregate is None:
            return peak_region_values
        else:
            raise ValueError(f"Invalid aggregate: {aggregate}. Must be 'sum', 'mean', or None")

    except Exception as e:
        warnings.warn(f"ATAC prediction failed: {e}")
        return None


def predict_rna(
    model,
    dna_sequence: str,
    ontology_terms: List[str],
    gene_start: int,
    gene_end: int,
    tss: int,
    sequence_length_bp: Optional[int] = None,
    aggregate: Optional[str] = 'sum',
    center_pad: bool = True,
    coord_transformer: Optional["CoordinateTransformer"] = None
) -> Union[np.ndarray, None]:
    """
    Predict RNA-seq signal from DNA sequence, aggregating over gene body only.

    Args:
        model: AlphaGenome DNA model
        dna_sequence: DNA sequence string
        ontology_terms: List of ontology term IDs (e.g., ['UBERON:0009834'])
        gene_start: Start coordinate of gene body (genomic coordinates)
        gene_end: End coordinate of gene body (genomic coordinates)
        tss: Transcription start site coordinate (genomic coordinates)
        sequence_length_bp: Expected sequence length (None = auto-detect)
        aggregate: 'sum', 'mean', or None (return full track)
        center_pad: Center and pad sequence with 'N's

    Returns:
        Aggregated values as array [encode+, encode-, gtex] if aggregate is not None,
        else full track array for gene body region only (shape: gene_length, 3)

    Note:
        The sequence should be centered on the TSS.
        Aggregation is performed ONLY over the gene body, not the entire sequence window.

    Example:
        >>> # Aggregate over gene body (20KB gene within 1MB sequence)
        >>> pred = predict_rna(
        ...     model, sequence, ['UBERON:0009834'],
        ...     gene_start=1040000, gene_end=1060000, tss=1041000
        ... )
        >>> print(f"RNA signal: encode+={pred[0]}, encode-={pred[1]}, gtex={pred[2]}")
    """
    try:
        from alphagenome.models import dna_client
    except ImportError:
        raise ImportError("AlphaGenome package not installed")

    # Center and pad/truncate sequence if requested
    if center_pad and sequence_length_bp is not None:
        current_length = len(dna_sequence)
        
        if current_length < sequence_length_bp:
            # Pad with N's to center the sequence
            dna_sequence = dna_sequence.center(sequence_length_bp, 'N')
        elif current_length > sequence_length_bp:
            # Truncate from both ends to center the sequence
            excess = current_length - sequence_length_bp
            trim_left = excess // 2
            trim_right = excess - trim_left
            dna_sequence = dna_sequence[trim_left:current_length - trim_right]
        # else: sequence is already the correct length, no changes needed

    try:
        # Make prediction
        output = model.predict_sequence(
            sequence=dna_sequence,
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=ontology_terms
        )

        # Extract RNA values
        rna_values = output.rna_seq.values  # Shape: (sequence_length, n_tracks)
        # Note: n_tracks = 3 for RNA: [encode+, encode-, gtex]

        # Transform coordinates if transformer provided (for personalized genomes)
        if coord_transformer is not None:
            gene_start, gene_end = coord_transformer.transform_interval(gene_start, gene_end)
            tss = coord_transformer.ref_to_personal(tss)

        # Calculate gene body region within sequence
        # Sequence is centered on TSS
        seq_center_idx = rna_values.shape[0] // 2
        
        # Calculate gene body boundaries relative to sequence center (TSS position)
        # If TSS is at seq_center_idx, then:
        # gene_start is (gene_start - tss) positions away from center
        # gene_end is (gene_end - tss) positions away from center
        start_idx = seq_center_idx + (gene_start - tss)
        end_idx = seq_center_idx + (gene_end - tss)
        
        # Clip to valid range
        start_idx = max(0, start_idx)
        end_idx = min(rna_values.shape[0], end_idx)
        
        # Extract gene body region only
        gene_body_values = rna_values[start_idx:end_idx, :]

        # Aggregate if requested
        if aggregate == 'sum':
            # Sum over sequence positions (axis=0), keep tracks (axis=1)
            # Returns array: [encode+, encode-, gtex]
            agg_values = gene_body_values.sum(axis=0)
            # Apply log1p to match log-transformed observed TPM values
            return np.log1p(agg_values)
        elif aggregate == 'mean':
            # Mean over sequence positions (axis=0), keep tracks (axis=1)
            agg_values = gene_body_values.mean(axis=0)
            # Apply log1p to match log-transformed observed TPM values
            return np.log1p(agg_values)
        elif aggregate is None:
            return gene_body_values
        else:
            raise ValueError(f"Invalid aggregate: {aggregate}. Must be 'sum', 'mean', or None")

    except Exception as e:
        warnings.warn(f"RNA prediction failed: {e}")
        return None


def predict_personal_genome_atac(
    model,
    maternal_sequence: str,
    paternal_sequence: str,
    ontology_terms: List[str],
    peak_start: int,
    peak_end: int,
    sequence_length_bp: Optional[int] = None,
    aggregate: Optional[str] = 'sum',
    center_pad: bool = True,
    maternal_transformer: Optional["CoordinateTransformer"] = None,
    paternal_transformer: Optional["CoordinateTransformer"] = None
) -> Union[float, np.ndarray]:
    """
    Predict ATAC-seq for personal genome (average of maternal and paternal).

    This is the standard pattern used across all inference scripts.

    Args:
        model: AlphaGenome DNA model
        maternal_sequence: Maternal haplotype sequence
        paternal_sequence: Paternal haplotype sequence
        ontology_terms: List of ontology term IDs
        peak_start: Start coordinate of peak region (genomic coordinates)
        peak_end: End coordinate of peak region (genomic coordinates)
        sequence_length_bp: Expected sequence length
        aggregate: 'sum', 'mean', or None
        center_pad: Center and pad sequences

    Returns:
        Average prediction (float or array) aggregated over peak region only

    Example:
        >>> pred = predict_personal_genome_atac(
        ...     model, mat_seq, pat_seq, ['EFO:0010843'],
        ...     peak_start=1045000, peak_end=1046500
        ... )
        >>> print(f"Personal ATAC signal in peak: {pred}")
    """
    # Predict for both haplotypes with coordinate transformation
    # Get raw arrays, aggregate each haplotype, average, then log1p once
    mat_pred = predict_atac(
        model, maternal_sequence, ontology_terms,
        peak_start, peak_end,
        sequence_length_bp, aggregate=None, center_pad=center_pad,
        coord_transformer=maternal_transformer
    )
    pat_pred = predict_atac(
        model, paternal_sequence, ontology_terms,
        peak_start, peak_end,
        sequence_length_bp, aggregate=None, center_pad=center_pad,
        coord_transformer=paternal_transformer
    )

    # Check for failures
    if mat_pred is None or pat_pred is None:
        return None

    # Aggregate each haplotype first (without log1p)
    if aggregate == 'sum':
        mat_agg = float(np.sum(mat_pred))
        pat_agg = float(np.sum(pat_pred))
    elif aggregate == 'mean':
        mat_agg = float(np.mean(mat_pred))
        pat_agg = float(np.mean(pat_pred))
    elif aggregate is None:
        # Handle shape mismatch for raw arrays
        if mat_pred.shape[0] != pat_pred.shape[0]:
            min_len = min(mat_pred.shape[0], pat_pred.shape[0])
            mat_pred = mat_pred[:min_len]
            pat_pred = pat_pred[:min_len]
        return (mat_pred + pat_pred) / 2
    else:
        raise ValueError(f"Invalid aggregate: {aggregate}")

    # Average the aggregated values, then apply log1p once
    avg_agg = (mat_agg + pat_agg) / 2
    return float(np.log1p(avg_agg))


def predict_atac_multi_tissue(
    model,
    dna_sequence: str,
    ontology_terms: List[str],
    peak_start: int,
    peak_end: int,
    sequence_length_bp: Optional[int] = None,
    aggregate: Optional[str] = 'sum',
    center_pad: bool = True,
    coord_transformer: Optional["CoordinateTransformer"] = None
) -> Dict[str, Union[float, None]]:
    """
    Predict ATAC-seq signal for multiple tissues in a single API call.
    
    This function efficiently predicts chromatin accessibility across multiple
    tissue types by making one API call with multiple ontology terms.
    
    Args:
        model: AlphaGenome DNA model
        dna_sequence: DNA sequence string
        ontology_terms: List of tissue ontology term IDs (e.g., ['EFO:0010843', 'EFO:0002067'])
        peak_start: Start coordinate of peak region (genomic coordinates)
        peak_end: End coordinate of peak region (genomic coordinates)
        sequence_length_bp: Expected sequence length (None = auto-detect)
        aggregate: 'sum', 'mean', or None (return full track)
        center_pad: Center and pad sequence with 'N's
    
    Returns:
        Dictionary mapping tissue_ontology -> prediction value (float or None if failed)
        Example: {'EFO:0010843': 5.2, 'EFO:0002067': 4.8}
    
    Note:
        - All tissues are predicted in a single API call (efficient)
        - Each tissue gets its own ATAC track in the output
        - Returns None for individual tissues that fail prediction
        - Predictions are log1p transformed to match observed data scale
    
    Example:
        >>> tissues = ['EFO:0010843', 'CL:0000236', 'CL:0000084']
        >>> pred_dict = predict_atac_multi_tissue(
        ...     model, sequence, tissues,
        ...     peak_start=1045000, peak_end=1046500
        ... )
        >>> for tissue, pred in pred_dict.items():
        ...     print(f"{tissue}: {pred}")
    """
    try:
        from alphagenome.models import dna_client
    except ImportError:
        raise ImportError("AlphaGenome package not installed")
    
    # Center and pad/truncate sequence if requested
    if center_pad and sequence_length_bp is not None:
        current_length = len(dna_sequence)
        
        if current_length < sequence_length_bp:
            dna_sequence = dna_sequence.center(sequence_length_bp, 'N')
        elif current_length > sequence_length_bp:
            excess = current_length - sequence_length_bp
            trim_left = excess // 2
            trim_right = excess - trim_left
            dna_sequence = dna_sequence[trim_left:current_length - trim_right]
    
    try:
        # Make prediction for all tissues at once
        output = model.predict_sequence(
            sequence=dna_sequence,
            requested_outputs=[dna_client.OutputType.ATAC],
            ontology_terms=ontology_terms
        )
        
        # Extract ATAC values: shape (sequence_length, n_tissues)
        atac_values = output.atac.values
        
        # Transform coordinates if transformer provided (for personalized genomes)
        if coord_transformer is not None:
            peak_start, peak_end = coord_transformer.transform_interval(peak_start, peak_end)
        
        # Calculate peak region within sequence
        seq_center_idx = atac_values.shape[0] // 2
        peak_center = (peak_start + peak_end) // 2
        peak_length = peak_end - peak_start
        
        start_idx = seq_center_idx - (peak_center - peak_start)
        end_idx = seq_center_idx + (peak_end - peak_center)
        
        # Clip to valid range
        start_idx = max(0, start_idx)
        end_idx = min(atac_values.shape[0], end_idx)
        
        # Extract peak region: shape (peak_length, n_tissues)
        peak_region_values = atac_values[start_idx:end_idx, :]
        
        # Create tissue -> prediction mapping
        tissue_predictions = {}
        
        for i, tissue in enumerate(ontology_terms):
            if i < peak_region_values.shape[1]:
                # Extract this tissue's track
                tissue_track = peak_region_values[:, i]
                
                # Aggregate if requested
                if aggregate == 'sum':
                    agg_value = float(np.sum(tissue_track))
                    # Apply log1p to match observed data
                    tissue_predictions[tissue] = np.log1p(agg_value)
                elif aggregate == 'mean':
                    agg_value = float(np.mean(tissue_track))
                    tissue_predictions[tissue] = np.log1p(agg_value)
                elif aggregate is None:
                    tissue_predictions[tissue] = tissue_track
                else:
                    raise ValueError(f"Invalid aggregate: {aggregate}")
            else:
                tissue_predictions[tissue] = None
        
        return tissue_predictions
        
    except Exception as e:
        warnings.warn(f"Multi-tissue ATAC prediction failed: {e}")
        # Return None for all tissues on failure
        return {tissue: None for tissue in ontology_terms}


def predict_personal_genome_atac_multi_tissue(
    model,
    maternal_sequence: str,
    paternal_sequence: str,
    ontology_terms: List[str],
    peak_start: int,
    peak_end: int,
    sequence_length_bp: Optional[int] = None,
    aggregate: Optional[str] = 'sum',
    center_pad: bool = True,
    maternal_transformer: Optional["CoordinateTransformer"] = None,
    paternal_transformer: Optional["CoordinateTransformer"] = None
) -> Dict[str, Union[float, None]]:
    """
    Predict ATAC-seq for multiple tissues on personal genome (maternal + paternal average).
    
    This is the standard pattern for multi-tissue inference, averaging predictions
    across both haplotypes for each tissue separately.
    
    Args:
        model: AlphaGenome DNA model
        maternal_sequence: Maternal haplotype sequence
        paternal_sequence: Paternal haplotype sequence
        ontology_terms: List of tissue ontology term IDs
        peak_start: Start coordinate of peak region
        peak_end: End coordinate of peak region
        sequence_length_bp: Expected sequence length
        aggregate: 'sum', 'mean', or None
        center_pad: Center and pad sequences
    
    Returns:
        Dictionary mapping tissue_ontology -> averaged prediction value
        Example: {'EFO:0010843': 5.2, 'EFO:0002067': 4.8}
    
    Note:
        - Makes TWO API calls (maternal + paternal), but each predicts all tissues
        - Averages maternal and paternal predictions per tissue
        - Returns None for tissues where either haplotype failed
    
    Example:
        >>> tissues = ['EFO:0010843', 'CL:0000236']
        >>> pred_dict = predict_personal_genome_atac_multi_tissue(
        ...     model, mat_seq, pat_seq, tissues,
        ...     peak_start=1045000, peak_end=1046500
        ... )
        >>> print(f"Tissue 1: {pred_dict['EFO:0010843']}")
    """
    # Predict for both haplotypes (each returns dict) with coordinate transformation
    # Get raw arrays, aggregate each, average, then log1p once
    mat_pred_dict = predict_atac_multi_tissue(
        model, maternal_sequence, ontology_terms,
        peak_start, peak_end,
        sequence_length_bp, aggregate=None, center_pad=center_pad,
        coord_transformer=maternal_transformer
    )

    pat_pred_dict = predict_atac_multi_tissue(
        model, paternal_sequence, ontology_terms,
        peak_start, peak_end,
        sequence_length_bp, aggregate=None, center_pad=center_pad,
        coord_transformer=paternal_transformer
    )

    # Aggregate each haplotype per tissue, average, then log1p once
    tissue_predictions = {}

    for tissue in ontology_terms:
        mat_pred = mat_pred_dict.get(tissue)
        pat_pred = pat_pred_dict.get(tissue)

        # Check for failures
        if mat_pred is None or pat_pred is None:
            tissue_predictions[tissue] = None
            continue

        # Aggregate each haplotype first (without log1p)
        if aggregate == 'sum':
            mat_agg = float(np.sum(mat_pred))
            pat_agg = float(np.sum(pat_pred))
        elif aggregate == 'mean':
            mat_agg = float(np.mean(mat_pred))
            pat_agg = float(np.mean(pat_pred))
        elif aggregate is None:
            # Handle shape mismatch for raw arrays
            if mat_pred.shape[0] != pat_pred.shape[0]:
                min_len = min(mat_pred.shape[0], pat_pred.shape[0])
                mat_pred = mat_pred[:min_len]
                pat_pred = pat_pred[:min_len]
            tissue_predictions[tissue] = (mat_pred + pat_pred) / 2
            continue
        else:
            raise ValueError(f"Invalid aggregate: {aggregate}")

        # Average the aggregated values, then apply log1p once
        avg_agg = (mat_agg + pat_agg) / 2
        tissue_predictions[tissue] = float(np.log1p(avg_agg))

    return tissue_predictions


def predict_personal_genome_rna(
    model,
    maternal_sequence: str,
    paternal_sequence: str,
    ontology_terms: List[str],
    gene_start: int,
    gene_end: int,
    tss: int,
    sequence_length_bp: Optional[int] = None,
    aggregate: Optional[str] = 'sum',
    center_pad: bool = True,
    maternal_transformer: Optional["CoordinateTransformer"] = None,
    paternal_transformer: Optional["CoordinateTransformer"] = None
) -> Union[Dict[str, float], None]:
    """
    Predict RNA-seq for personal genome (average of maternal and paternal).

    Args:
        model: AlphaGenome DNA model
        maternal_sequence: Maternal haplotype sequence
        paternal_sequence: Paternal haplotype sequence
        ontology_terms: List of ontology term IDs
        gene_start: Start coordinate of gene body (genomic coordinates)
        gene_end: End coordinate of gene body (genomic coordinates)
        tss: Transcription start site coordinate (genomic coordinates)
        sequence_length_bp: Expected sequence length
        aggregate: 'sum', 'mean', or None
        center_pad: Center and pad sequences
        maternal_transformer: Optional coordinate transformer for maternal haplotype
        paternal_transformer: Optional coordinate transformer for paternal haplotype

    Returns:
        Dictionary with keys:
            - 'encode_combined': Sum of encode+ and encode- strands (from averaged haplotypes)
            - 'gtex': GTEx reference prediction (from averaged haplotypes)
        Returns None if prediction fails

    Example:
        >>> pred = predict_personal_genome_rna(
        ...     model, mat_seq, pat_seq, ['UBERON:0009834'],
        ...     gene_start=1040000, gene_end=1060000, tss=1041000
        ... )
        >>> print(pred)
        {'encode_combined': 1234.5, 'gtex': 2345.6}
    """
    # Predict for both haplotypes
    mat_pred = predict_rna(
        model, maternal_sequence, ontology_terms,
        gene_start, gene_end, tss,
        sequence_length_bp, aggregate=aggregate, center_pad=center_pad,
        coord_transformer=maternal_transformer
    )
    pat_pred = predict_rna(
        model, paternal_sequence, ontology_terms,
        gene_start, gene_end, tss,
        sequence_length_bp, aggregate=aggregate, center_pad=center_pad,
        coord_transformer=paternal_transformer
    )

    # Check for failures
    if mat_pred is None or pat_pred is None:
        return None

    # Average maternal and paternal predictions
    # avg_pred shape: [encode+, encode-, gtex] (3 tracks)
    avg_pred = (mat_pred + pat_pred) / 2

    # Return dictionary with two tracks
    # encode_combined = SUM of both strands (encode+ + encode-)
    return {
        'encode_combined': float(avg_pred[0] + avg_pred[1]),
        'gtex': float(avg_pred[2])
    }


def aggregate_over_region(
    predictions: np.ndarray,
    region_start: int,
    region_end: int,
    tss: int,
    sequence_length_bp: int,
    method: str = 'sum',
    log_transform: bool = False
) -> np.ndarray:
    """
    Aggregate prediction track over a genomic region (peak or gene body).

    Args:
        predictions: Array of shape (sequence_length, n_tracks)
        region_start: Genomic start coordinate (bp)
        region_end: Genomic end coordinate (bp)
        tss: Center position (TSS or peak center)
        sequence_length_bp: Length of prediction sequence
        method: 'sum' or 'mean'
        log_transform: Apply log1p transform (useful for ATAC)

    Returns:
        Aggregated values of shape (n_tracks,)

    Example:
        >>> track = model.predict_sequence(...)  # Returns full track
        >>> agg = aggregate_over_region(
        ...     track, peak_start, peak_end, peak_center, 100000
        ... )
        >>> print(f"Aggregated signal: {agg}")
    """
    # Convert genomic coordinates to sequence indices
    seq_center_idx = sequence_length_bp // 2

    # Calculate start and end indices in sequence
    start_idx = seq_center_idx + (region_start - tss)
    end_idx = seq_center_idx + (region_end - tss)

    # Clip to valid range
    start_idx = max(0, start_idx)
    end_idx = min(sequence_length_bp, end_idx)

    # Extract region
    if end_idx > start_idx:
        region_predictions = predictions[start_idx:end_idx, :]
    else:
        # Fallback: use entire prediction
        warnings.warn(
            f"Invalid region indices ({start_idx}, {end_idx}). "
            "Using entire sequence."
        )
        region_predictions = predictions

    # Aggregate
    if method == 'sum':
        agg = region_predictions.sum(axis=0)
    elif method == 'mean':
        agg = region_predictions.mean(axis=0)
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'sum' or 'mean'")

    # Optional log transform
    if log_transform:
        agg = np.log1p(agg)

    return agg


def batch_predict_regions(
    model,
    sequences: List[Tuple[str, str]],
    ontology_terms: List[str],
    prediction_type: Literal['atac', 'rna'] = 'atac',
    sequence_length_bp: Optional[int] = None,
    aggregate: str = 'sum',
    progress_callback: Optional[callable] = None
) -> List[Optional[float]]:
    """
    Batch prediction for multiple regions/samples.

    Args:
        model: AlphaGenome DNA model
        sequences: List of (maternal_seq, paternal_seq) tuples
        ontology_terms: Ontology terms for predictions
        prediction_type: 'atac' or 'rna'
        sequence_length_bp: Expected sequence length
        aggregate: Aggregation method
        progress_callback: Optional function(current, total) for progress

    Returns:
        List of predictions (None for failed predictions)

    Example:
        >>> sequences = [(mat1, pat1), (mat2, pat2), (mat3, pat3)]
        >>> predictions = batch_predict_regions(
        ...     model, sequences, ['EFO:0010843'], prediction_type='atac'
        ... )
        >>> print(f"Completed {len([p for p in predictions if p])} predictions")
    """
    predictions = []

    predict_func = (
        predict_personal_genome_atac if prediction_type == 'atac'
        else predict_personal_genome_rna
    )

    for i, (mat_seq, pat_seq) in enumerate(sequences):
        pred = predict_func(
            model, mat_seq, pat_seq, ontology_terms,
            sequence_length_bp, aggregate
        )
        predictions.append(pred)

        # Progress callback
        if progress_callback is not None:
            progress_callback(i + 1, len(sequences))

    return predictions


def calculate_prediction_statistics(
    predictions: List[float],
    observations: List[float]
) -> dict:
    """
    Calculate statistics comparing predictions to observations.

    Args:
        predictions: List of predicted values
        observations: List of observed values

    Returns:
        Dict with correlation, R², MAE, RMSE

    Example:
        >>> stats = calculate_prediction_statistics(preds, obs)
        >>> print(f"Pearson r: {stats['pearson']:.3f}")
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    predictions = np.array(predictions)
    observations = np.array(observations)

    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(observations))
    predictions = predictions[mask]
    observations = observations[mask]

    if len(predictions) < 2:
        return {
            'pearson': np.nan,
            'pearson_pvalue': np.nan,
            'r2': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'n_samples': len(predictions)
        }

    # Calculate statistics
    pearson_r, pearson_p = pearsonr(predictions, observations)
    r2 = r2_score(observations, predictions)
    mae = mean_absolute_error(observations, predictions)
    rmse = np.sqrt(mean_squared_error(observations, predictions))

    return {
        'pearson': pearson_r,
        'pearson_pvalue': pearson_p,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(predictions)
    }


def infer_prediction_type_from_ontology(ontology_terms: List[str]) -> str:
    """
    Infer whether to use ATAC or RNA prediction based on ontology terms.

    Args:
        ontology_terms: List of ontology term IDs

    Returns:
        'atac' or 'rna'

    Example:
        >>> pred_type = infer_prediction_type_from_ontology(['EFO:0010843'])
        >>> print(pred_type)  # 'atac'
    """
    # Simple heuristic: EFO terms are typically cell lines (ATAC)
    # UBERON terms are typically tissues (RNA)
    if any(term.startswith('EFO:') for term in ontology_terms):
        return 'atac'
    elif any(term.startswith('UBERON:') for term in ontology_terms):
        return 'rna'
    else:
        # Default to ATAC
        warnings.warn(
            f"Could not infer prediction type from ontology terms {ontology_terms}. "
            "Defaulting to 'atac'"
        )
        return 'atac'