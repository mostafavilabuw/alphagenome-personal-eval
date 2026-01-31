"""
Borzoi model utilities for AlphaGenome evaluation.

This module provides functions to interact with the Borzoi model, including:
- Model initialization (using AnnotatedBorzoi or Flashzoi for faster inference)
- Sequence processing (one-hot encoding)
- Prediction and region aggregation
- Personal genome prediction (maternal + paternal averaging)
- Track index search utilities
- ATAC-seq/DNase chromatin accessibility prediction (single and multi-tissue)
- RNA-seq expression prediction

ATAC-seq Support:
    Borzoi uses DNASE tracks for chromatin accessibility prediction.
    Functions like `predict_borzoi_atac()` and `predict_borzoi_personal_atac()`
    mirror the AlphaGenome API pattern for easy comparison.
    
    Multi-tissue prediction is supported via:
    - `predict_borzoi_atac_multi_tissue()` - single sequence, multiple tissues
    - `predict_borzoi_personal_atac_multi_tissue()` - personal genome, multiple tissues

Flashzoi Support:
    Flashzoi is a FlashAttention-2 optimized version of Borzoi that provides:
    - ~1.6x faster inference speed
    - ~18% memory reduction
    Use `use_flashzoi=True` in `init_borzoi_model()` for faster inference.
"""

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any, Literal, TYPE_CHECKING

from .gpu import select_device

if TYPE_CHECKING:
    from alphagenome_eval.utils.coordinates import CoordinateTransformer

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from borzoi_pytorch import Borzoi, AnnotatedBorzoi
except ImportError:
    Borzoi = None
    AnnotatedBorzoi = None


# =============================================================================
# Fast One-Hot Encoding (Numba-accelerated)
# Credit: tangermeme library (https://github.com/jmschrei/tangermeme)
# =============================================================================

if HAS_NUMBA:
    @numba.njit("void(int8[:, :], int8[:], int8[:])")
    def _fast_one_hot_encode(X_ohe, seq, mapping):
        """JIT-compiled one-hot encoding inner loop."""
        for i in range(len(seq)):
            idx = mapping[seq[i]]
            if idx == -1:  # ignored character (N)
                continue
            if idx == -2:
                raise ValueError("Encountered unknown character")
            X_ohe[i, idx] = 1


def _onehot_encoding_numba(sequence: str, alphabet: str = "ACGT", 
                           dtype: torch.dtype = torch.float32, 
                           ignore: str = "N",
                           neutral_value: float = 0.0) -> torch.Tensor:
    """
    Fast one-hot encoding using numba JIT compilation.
    
    Args:
        sequence: DNA sequence string (will be uppercased)
        alphabet: Characters to encode (default: "ACGT")
        dtype: Output tensor dtype (default: torch.float32)
        ignore: Characters to ignore (default: "N")
        neutral_value: Value to fill for ignored characters (default: 0.0).
                       Use 0.25 for uniform distribution across bases.
        
    Returns:
        Tensor of shape (len(alphabet), len(sequence))
    """
    if not HAS_NUMBA:
        raise ImportError("numba is required for fast one-hot encoding. "
                         "Install with: pip install numba")
    
    seq_idxs = np.frombuffer(bytearray(sequence.upper(), "utf8"), dtype=np.int8)
    alpha_idxs = np.frombuffer(bytearray(alphabet, "utf8"), dtype=np.int8)
    ignore_idxs = np.frombuffer(bytearray(ignore, "utf8"), dtype=np.int8)

    one_hot_mapping = np.zeros(256, dtype=np.int8) - 2
    for i, idx in enumerate(alpha_idxs):
        one_hot_mapping[idx] = i
    for idx in ignore_idxs:
        one_hot_mapping[idx] = -1

    one_hot = np.zeros((len(sequence), len(alphabet)), dtype=np.int8)
    _fast_one_hot_encode(one_hot, seq_idxs, one_hot_mapping)
    
    # Convert to float and apply neutral_value for ignored characters
    one_hot_float = one_hot.astype(np.float32)
    if neutral_value != 0.0:
        # Find positions where all values are zero (ignored characters)
        ignored_mask = one_hot_float.sum(axis=1) == 0
        one_hot_float[ignored_mask, :] = neutral_value
    
    return torch.from_numpy(one_hot_float).type(dtype).T


# =============================================================================
# Borzoi Model Constants
# =============================================================================

BORZOI_SEQ_LEN = 524288       # Input sequence length (524kb)
BORZOI_BIN_SIZE = 32          # Output bin size in bp
BORZOI_NUM_BINS = 6144        # Number of output bins
BORZOI_NUM_TRACKS = 7611      # Total number of output tracks
BORZOI_PRED_LENGTH = 196608   # Prediction coverage: 6144 * 32 bp (~196kb)

# Supported context window sizes for Borzoi inference
# Smaller windows = faster VCF parsing, but sequences are center-padded with 'N'
# Note: The model always receives 524KB input; smaller context windows are padded
BORZOI_SUPPORTED_CONTEXT_WINDOWS = [
    4096,    # 4KB
    8192,    # 8KB
    16384,   # 16KB
    65536,   # 64KB
    102400,  # 100KB
    131072,  # 128KB
    262144,  # 256KB
    524288,  # 524KB (full, default)
]

# Model name mappings for different replicates
BORZOI_MODEL_NAMES = {
    0: 'johahi/borzoi-replicate-0',
    1: 'johahi/borzoi-replicate-1',
    2: 'johahi/borzoi-replicate-2',
    3: 'johahi/borzoi-replicate-3',
}

FLASHZOI_MODEL_NAMES = {
    0: 'johahi/flashzoi-replicate-0',
    1: 'johahi/flashzoi-replicate-1',
    2: 'johahi/flashzoi-replicate-2',
    3: 'johahi/flashzoi-replicate-3',
}

# Default model names
DEFAULT_BORZOI_MODEL = 'johahi/borzoi-replicate-0'
DEFAULT_FLASHZOI_MODEL = 'johahi/flashzoi-replicate-0'


def validate_borzoi_context_window(context_window: int) -> int:
    """
    Validate that the context window size is supported for Borzoi inference.
    
    Borzoi requires a fixed 524KB input sequence. When using smaller context windows,
    the extracted sequence is center-padded with 'N' nucleotides to reach 524KB.
    
    Args:
        context_window: Desired context window size in base pairs
        
    Returns:
        Validated context window size
        
    Raises:
        ValueError: If context_window is not in BORZOI_SUPPORTED_CONTEXT_WINDOWS
        
    Example:
        >>> validate_borzoi_context_window(16384)  # 16KB
        16384
        >>> validate_borzoi_context_window(524288)  # Full 524KB
        524288
        >>> validate_borzoi_context_window(10000)  # Invalid
        ValueError: Invalid context window size: 10000. Supported sizes: ...
    """
    if context_window not in BORZOI_SUPPORTED_CONTEXT_WINDOWS:
        supported_str = ', '.join(f"{s} ({s//1024}KB)" for s in BORZOI_SUPPORTED_CONTEXT_WINDOWS)
        raise ValueError(
            f"Invalid context window size: {context_window}. "
            f"Supported sizes: {supported_str}"
        )
    return context_window


def get_model_name(
    replicate: int = 0,
    use_flashzoi: bool = False
) -> str:
    """
    Get the HuggingFace model name for a Borzoi/Flashzoi replicate.
    
    Args:
        replicate: Model replicate number (0-3)
        use_flashzoi: If True, return Flashzoi model name (FlashAttention-2 optimized)
        
    Returns:
        HuggingFace model name string
        
    Example:
        >>> get_model_name(0, use_flashzoi=False)
        'johahi/borzoi-replicate-0'
        >>> get_model_name(0, use_flashzoi=True)
        'johahi/flashzoi-replicate-0'
    """
    if replicate not in range(4):
        raise ValueError(f"replicate must be 0-3, got {replicate}")
    
    if use_flashzoi:
        return FLASHZOI_MODEL_NAMES[replicate]
    return BORZOI_MODEL_NAMES[replicate]


def init_borzoi_model(
    model_name: Optional[str] = None,
    device: str = 'auto',
    use_flashzoi: bool = False,
    replicate: int = 0,
    use_random_init: bool = False
) -> Tuple[Any, Optional[pd.DataFrame]]:
    """
    Initialize Borzoi or Flashzoi model and load track metadata.

    Args:
        model_name: HuggingFace model name (if provided, overrides use_flashzoi and replicate)
        device: Device to load model on. Options:
                - 'auto': Automatically select best GPU (most free memory) or CPU
                - 'cuda': Use default CUDA device
                - 'cuda:N': Use specific GPU N
                - 'cpu': Use CPU
        use_flashzoi: If True, use Flashzoi (FlashAttention-2 optimized) for faster inference.
                      Provides ~1.6x speedup and ~18% memory reduction.
        replicate: Model replicate number (0-3), used when model_name is None
        use_random_init: If True, initialize model with random weights instead of loading
                         pretrained weights. Useful for baseline comparisons.

    Returns:
        Tuple of (model, tracks_df)
        - model: Loaded Borzoi/Flashzoi model (eval mode)
        - tracks_df: DataFrame with track metadata (if available)
        
    Example:
        >>> # Load standard Borzoi (auto-selects best GPU)
        >>> model, tracks_df = init_borzoi_model()
        
        >>> # Load Flashzoi for faster inference
        >>> model, tracks_df = init_borzoi_model(use_flashzoi=True)
        
        >>> # Load specific replicate
        >>> model, tracks_df = init_borzoi_model(use_flashzoi=True, replicate=1)
        
        >>> # Load specific model by name
        >>> model, tracks_df = init_borzoi_model(model_name='johahi/flashzoi-replicate-0')
        
        >>> # Force CPU
        >>> model, tracks_df = init_borzoi_model(device='cpu')

        >>> # Random initialization (no pretrained weights)
        >>> model, tracks_df = init_borzoi_model(use_flashzoi=True, use_random_init=True)

    Note:
        Flashzoi uses FlashAttention-2 which requires:
        - CUDA GPU with compute capability >= 8.0 (Ampere or newer)
        - flash-attn package installed
        
        Performance comparison (on NVIDIA RTX A4000):
        - Standard Borzoi: ~0.24s per inference, ~3.5 GB memory
        - Flashzoi: ~0.15s per inference, ~2.9 GB memory
    """
    if AnnotatedBorzoi is None:
        raise ImportError("borzoi_pytorch not installed. Please install it to use Borzoi models.")

    # Auto-select device if requested
    if device == 'auto':
        device = select_device(prefer_cuda=True)
        print(f"Auto-selected device: {device}")

    # Handle random initialization (no pretrained weights)
    if use_random_init:
        return _init_borzoi_random(device, use_flashzoi, replicate)

    # Determine model name
    if model_name is None:
        model_name = get_model_name(replicate=replicate, use_flashzoi=use_flashzoi)
    
    # Determine model variant for logging
    is_flashzoi = 'flashzoi' in model_name.lower()
    variant_name = "Flashzoi" if is_flashzoi else "Borzoi"
    
    try:
        # Try loading AnnotatedBorzoi first to get metadata
        model = AnnotatedBorzoi.from_pretrained(model_name)
        tracks_df = model.tracks_df.copy()
        print(f"Loaded {variant_name} model (AnnotatedBorzoi) with {len(tracks_df)} tracks")
        if is_flashzoi:
            print("  → Using FlashAttention-2 for faster inference (~1.6x speedup)")
    except Exception as e:
        print(f"Could not load AnnotatedBorzoi ({e}), falling back to Borzoi class")
        model = Borzoi.from_pretrained(model_name)
        tracks_df = None
        print(f"Loaded {variant_name} model (Borzoi) without track metadata")
        if is_flashzoi:
            print("  → Using FlashAttention-2 for faster inference (~1.6x speedup)")

    model = model.to(device)
    model.eval()

    return model, tracks_df


def _init_borzoi_random(
    device: str,
    use_flashzoi: bool,
    replicate: int
) -> Tuple[Any, Optional[pd.DataFrame]]:
    """
    Initialize Borzoi/Flashzoi with random weights (no pretrained loading).

    This creates the model architecture with random PyTorch initialization,
    without downloading any pretrained weights from HuggingFace.

    Args:
        device: Device to load model on
        use_flashzoi: If True, use FlashAttention-2 architecture
        replicate: Model replicate number (for track metadata URL)

    Returns:
        Tuple of (model, tracks_df)
    """
    from borzoi_pytorch.config_borzoi import BorzoiConfig

    # Create config - set flashed=True for FlashAttention-2
    config = BorzoiConfig(
        flashed=use_flashzoi,  # True = FlashAttention, False = regular attention
        enable_mouse_head=True,  # Match pretrained config
    )

    # Direct instantiation - NO weights loaded!
    model = Borzoi(config)
    model = model.to(device)
    model.eval()

    variant = "Flashzoi" if use_flashzoi else "Borzoi"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Initialized RANDOM {variant} (no pretrained weights)")
    print(f"  flashed={use_flashzoi}, params={n_params:.1f}M")

    # Load tracks_df from borzoi_pytorch package (no network required)
    tracks_df = None
    try:
        from borzoi_pytorch.pytorch_borzoi_model import TRACKS_DF
        tracks_df = TRACKS_DF.copy()
        print(f"Loaded track metadata ({len(tracks_df)} tracks)")
    except Exception as e:
        print(f"Could not load track metadata: {e}")

    return model, tracks_df


def dna_to_onehot(
    sequence: str,
    seq_len: int = 524288,
    neutral_value: float = 0.0
) -> torch.Tensor:
    """
    Convert DNA sequence string to one-hot encoded tensor for Borzoi.
    
    Uses numba-accelerated encoding for fast performance.
    
    Args:
        sequence: DNA sequence string
        seq_len: Target sequence length (default 524288 for Borzoi)
        neutral_value: Value to fill for N bases (default: 0.0).
                       Use 0.25 for uniform distribution across bases.
        
    Returns:
        Tensor of shape (1, 4, seq_len)
        
    Raises:
        ImportError: If numba is not installed
    """
    if not HAS_NUMBA:
        raise ImportError("numba is required for one-hot encoding. "
                         "Install with: pip install numba")
    
    # Pad or truncate to target length
    current_len = len(sequence)
    if current_len < seq_len:
        # Center pad with N
        pad_left = (seq_len - current_len) // 2
        pad_right = seq_len - current_len - pad_left
        sequence = 'N' * pad_left + sequence + 'N' * pad_right
    elif current_len > seq_len:
        # Center crop
        start = (current_len - seq_len) // 2
        sequence = sequence[start : start + seq_len]
    
    one_hot = _onehot_encoding_numba(sequence, dtype=torch.float32, neutral_value=neutral_value)
    return one_hot.unsqueeze(0)  # (1, 4, L)


def predict_borzoi(
    model: Any,
    sequence: str,
    target_tracks: Optional[List[int]] = None,
    device: str = 'cuda',
    seq_len: int = BORZOI_SEQ_LEN
) -> np.ndarray:
    """
    Run Borzoi prediction on a sequence.

    Args:
        model: Loaded Borzoi model
        sequence: DNA sequence string
        target_tracks: List of track indices to return (None = return all)
        device: Device to run on
        seq_len: Sequence length expected by model (default: 524288)

    Returns:
        Numpy array of predictions.
        Shape: (n_tracks, n_bins) where n_bins=6144
        - If target_tracks is None: (7611, 6144)
        - If target_tracks provided: (len(target_tracks), 6144)
        
    Note:
        Borzoi outputs (batch, tracks, bins) = (1, 7611, 6144)
        Each bin covers 32bp, total coverage = 196,608bp (center of input)
    """
    # Prepare input
    input_tensor = dna_to_onehot(sequence, seq_len).to(device)
    
    # Run inference
    with torch.no_grad():
        # Use autocast for efficiency if available
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(input_tensor)
        else:
            output = model(input_tensor)
            
    # Output shape: (Batch, Tracks, Bins) -> (1, 7611, 6144)
    # Squeeze batch dimension -> (7611, 6144)
    output = output.squeeze(0)
    
    # OPTIMIZATION: Select specific tracks on GPU BEFORE dtype conversion and CPU transfer
    # This reduces both computation (only convert subset) and data transfer (~200MB -> ~400KB)
    if target_tracks is not None:
        output = output[target_tracks, :].to(dtype=torch.float32)
    else:
        output = output.to(dtype=torch.float32)
    
    # Transfer to CPU and convert to numpy
    output = output.cpu().numpy()
        
    return output


def get_borzoi_track_indices(
    tracks_df: pd.DataFrame,
    description_query: str,
    assay_type: Optional[str] = None
) -> List[int]:
    """
    Find track indices matching a description query.

    Args:
        tracks_df: DataFrame with track metadata
        description_query: String to search for in description (case-insensitive)
        assay_type: Optional filter for assay type (e.g., 'DNASE', 'RNA', 'CAGE')

    Returns:
        List of track indices
    """
    if tracks_df is None:
        return []
        
    mask = tracks_df['description'].str.contains(description_query, case=False, na=False)
    
    if assay_type:
        # Assay type is typically at the beginning of description (e.g., "RNA:", "CAGE:", "DNASE:")
        if 'assay' in tracks_df.columns:
            mask &= (tracks_df['assay'] == assay_type)
        else:
            # Check if description starts with assay type prefix
            assay_prefix = f"{assay_type}:"
            mask &= tracks_df['description'].str.startswith(assay_prefix, na=False)
            
    return tracks_df[mask].index.tolist()


def get_borzoi_dnase_track_indices(
    tracks_df: pd.DataFrame,
    tissue_query: Optional[str] = None,
    exclude_disease: bool = True,
    exclude_embryonic: bool = True
) -> List[int]:
    """
    Find DNASE/ATAC track indices, optionally filtered by tissue type.
    
    Borzoi uses DNASE tracks for chromatin accessibility prediction.
    This function provides convenient filtering for ATAC-seq style analyses.

    Args:
        tracks_df: DataFrame with track metadata (from AnnotatedBorzoi)
        tissue_query: Optional tissue keyword to filter (e.g., 'brain', 'B cell', 'GM12878')
        exclude_disease: Exclude disease-related samples
        exclude_embryonic: Exclude embryonic/fetal samples

    Returns:
        List of track indices for DNASE tracks
        
    Example:
        >>> # Get all DNASE tracks
        >>> all_dnase = get_borzoi_dnase_track_indices(tracks_df)
        
        >>> # Get brain DNASE tracks (healthy adult)
        >>> brain_dnase = get_borzoi_dnase_track_indices(tracks_df, tissue_query='brain')
        
        >>> # Get LCL/B-cell tracks
        >>> lcl_dnase = get_borzoi_dnase_track_indices(tracks_df, tissue_query='GM12878')
    """
    if tracks_df is None:
        return []
    
    # Start with DNASE tracks only
    mask = tracks_df['description'].str.startswith('DNASE:', na=False)
    
    # Filter by tissue if provided
    if tissue_query:
        tissue_mask = tracks_df['description'].str.contains(tissue_query, case=False, na=False)
        mask &= tissue_mask
    
    # Exclude disease-related samples
    if exclude_disease:
        disease_keywords = ['Alzheimer', 'cognitive', 'impair', 'disease', 'disorder', 
                          'pathology', 'tumor', 'cancer', 'leukemia', 'carcinoma']
        disease_pattern = '|'.join(disease_keywords)
        not_disease = ~tracks_df['description'].str.contains(disease_pattern, case=False, na=False)
        mask &= not_disease
    
    # Exclude embryonic/fetal samples
    if exclude_embryonic:
        embryo_keywords = ['embryo', 'fetal', 'fetus', 'embryonic']
        embryo_pattern = '|'.join(embryo_keywords)
        not_embryo = ~tracks_df['description'].str.contains(embryo_pattern, case=False, na=False)
        mask &= not_embryo
    
    return tracks_df[mask].index.tolist()


# =============================================================================
# Region Aggregation
# =============================================================================

def aggregate_borzoi_over_region(
    predictions: np.ndarray,
    region_start: int,
    region_end: int,
    seq_center: int,
    method: str = 'sum',
    log_transform: bool = True,
    coord_transformer: Optional["CoordinateTransformer"] = None
) -> np.ndarray:
    """
    Aggregate Borzoi predictions over a genomic region (gene body or peak).
    
    Args:
        predictions: Borzoi output array, shape (n_tracks, n_bins) or (n_bins,)
        region_start: Start of region in genomic coordinates (1-based)
        region_end: End of region in genomic coordinates (1-based)
        seq_center: Center position of the input sequence (genomic coordinates)
        method: Aggregation method ('sum', 'mean', or 'max')
        log_transform: Apply log1p transformation after aggregation
        coord_transformer: Optional coordinate transformer for personalized genomes
        
    Returns:
        Aggregated values per track, shape (n_tracks,) or scalar if single track
        
    Note:
        - Borzoi output has BORZOI_NUM_BINS (6144) bins at BORZOI_BIN_SIZE (32bp) resolution
        - Output covers center BORZOI_PRED_LENGTH (196,608bp) of input sequence
        - Center bin (3072) aligns with seq_center
        
    Example:
        >>> pred = predict_borzoi(model, sequence)  # (7611, 6144)
        >>> gene_pred = aggregate_borzoi_over_region(
        ...     pred, gene_start=1000000, gene_end=1050000, 
        ...     seq_center=1025000, method='sum'
        ... )
        >>> print(gene_pred.shape)  # (7611,)
    """
    # Handle 1D input (single track)
    single_track = predictions.ndim == 1
    if single_track:
        predictions = predictions.reshape(1, -1)
    
    num_bins = predictions.shape[1]  # Should be 6144
    
    # Transform coordinates if transformer provided (for personalized genomes)
    if coord_transformer is not None:
        region_start, region_end = coord_transformer.transform_interval(region_start, region_end)
    
    # Calculate prediction boundaries
    # The prediction covers center BORZOI_PRED_LENGTH of the input
    pred_half_length = BORZOI_PRED_LENGTH // 2
    pred_start = seq_center - pred_half_length
    pred_end = seq_center + pred_half_length
    
    # Map region coordinates to bin indices
    # Use floor for start (inclusive start position) and ceil for end (inclusive end position)
    region_start_rel = region_start - pred_start
    region_end_rel = region_end - pred_start
    
    start_bin = max(0, region_start_rel // BORZOI_BIN_SIZE)
    end_bin = min(num_bins, int(np.ceil(region_end_rel / BORZOI_BIN_SIZE)))
    
    # Handle edge cases
    if start_bin >= end_bin or start_bin >= num_bins or end_bin <= 0:
        # Region doesn't overlap with predictions
        result = np.zeros(predictions.shape[0])
        if log_transform:
            result = np.log1p(result)
        return result[0] if single_track else result
    
    # Extract bins overlapping the region
    region_preds = predictions[:, start_bin:end_bin]
    
    # Aggregate
    if method == 'sum':
        aggregated = region_preds.sum(axis=1)
    elif method == 'mean':
        aggregated = region_preds.mean(axis=1)
    elif method == 'max':
        aggregated = region_preds.max(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Use 'sum', 'mean', or 'max'")
    
    # Optional log transform
    if log_transform:
        aggregated = np.log1p(aggregated)
    
    return aggregated[0] if single_track else aggregated


# =============================================================================
# Personal Genome Prediction
# =============================================================================

def predict_borzoi_personal_genome(
    model: Any,
    maternal_sequence: str,
    paternal_sequence: str,
    target_tracks: Optional[List[int]] = None,
    device: str = 'cuda',
    region_start: Optional[int] = None,
    region_end: Optional[int] = None,
    seq_center: Optional[int] = None,
    aggregate_method: str = 'sum',
    log_transform: bool = True,
    maternal_transformer: Optional["CoordinateTransformer"] = None,
    paternal_transformer: Optional["CoordinateTransformer"] = None
) -> Union[np.ndarray, float]:
    """
    Run Borzoi prediction on personal genome (average of maternal and paternal).
    
    This function predicts on both haplotypes and averages the results,
    optionally aggregating over a specific genomic region.
    
    Args:
        model: Loaded Borzoi model
        maternal_sequence: Maternal haplotype DNA sequence
        paternal_sequence: Paternal haplotype DNA sequence
        target_tracks: List of track indices to return (None = all tracks)
        device: Device to run on
        region_start: Start of region for aggregation (genomic coords, optional)
        region_end: End of region for aggregation (genomic coords, optional)
        seq_center: Center position of sequence (genomic coords, required if aggregating)
        aggregate_method: 'sum', 'mean', or 'max' (only used if region specified)
        log_transform: Apply log1p after aggregation
        maternal_transformer: Optional coordinate transformer for maternal haplotype
        paternal_transformer: Optional coordinate transformer for paternal haplotype
        
    Returns:
        If region specified: Aggregated values per track, shape (n_tracks,)
        If no region: Raw predictions, shape (n_tracks, n_bins)
        
    Example:
        >>> # Full prediction (no aggregation)
        >>> pred = predict_borzoi_personal_genome(
        ...     model, mat_seq, pat_seq, target_tracks=brain_indices
        ... )
        >>> print(pred.shape)  # (29, 6144) for 29 brain tracks
        
        >>> # Aggregated over gene body
        >>> gene_pred = predict_borzoi_personal_genome(
        ...     model, mat_seq, pat_seq,
        ...     target_tracks=brain_indices,
        ...     region_start=gene_start, region_end=gene_end,
        ...     seq_center=tss, aggregate_method='sum'
        ... )
        >>> print(gene_pred.shape)  # (29,)
    """
    # Predict on both haplotypes
    mat_pred = predict_borzoi(model, maternal_sequence, target_tracks, device)
    pat_pred = predict_borzoi(model, paternal_sequence, target_tracks, device)
    
    # Aggregate per haplotype if specified (needed for asymmetric indels)
    if region_start is not None and region_end is not None:
        if seq_center is None:
            raise ValueError("seq_center required when specifying region for aggregation")
        
        # Aggregate maternal and paternal separately with their respective transformers
        mat_agg = aggregate_borzoi_over_region(
            mat_pred,
            region_start=region_start,
            region_end=region_end,
            seq_center=seq_center,
            method=aggregate_method,
            log_transform=False,  # Don't log yet
            coord_transformer=maternal_transformer
        )
        
        pat_agg = aggregate_borzoi_over_region(
            pat_pred,
            region_start=region_start,
            region_end=region_end,
            seq_center=seq_center,
            method=aggregate_method,
            log_transform=False,  # Don't log yet
            coord_transformer=paternal_transformer
        )
        
        # Average aggregated predictions
        avg_pred = (mat_agg + pat_agg) / 2
        
        # Apply log transform if requested
        if log_transform:
            avg_pred = np.log1p(avg_pred)
    else:
        # No aggregation: just average raw predictions
        avg_pred = (mat_pred + pat_pred) / 2
    
    return avg_pred


def predict_borzoi_rna(
    model: Any,
    sequence: str,
    track_indices: List[int],
    gene_start: int,
    gene_end: int,
    tss: int,
    device: str = 'cuda',
    aggregate_method: str = 'sum',
    log_transform: bool = True
) -> Dict[str, float]:
    """
    Predict RNA expression using Borzoi with specified tracks.
    
    Runs prediction on specified tracks and aggregates over the gene body.
    
    Args:
        model: Loaded Borzoi model
        sequence: DNA sequence (centered on TSS)
        track_indices: List of track indices to use for prediction
        gene_start: Gene body start (genomic coords)
        gene_end: Gene body end (genomic coords)
        tss: Transcription start site (center of input sequence)
        device: Device to run on
        aggregate_method: Aggregation method ('sum', 'mean', 'max')
        log_transform: Apply log1p transformation
        
    Returns:
        Dictionary with:
        - 'mean': Mean prediction across tracks
        - 'std': Standard deviation across tracks
        - 'n_tracks': Number of tracks used
        - 'track_values': Array of per-track predictions
        
    Example:
        >>> # First, select tracks in notebook/script
        >>> brain_mask = tracks_df['description'].str.contains('brain', case=False)
        >>> rna_mask = tracks_df['description'].str.startswith('RNA:')
        >>> track_indices = tracks_df[brain_mask & rna_mask].index.tolist()
        >>> 
        >>> result = predict_borzoi_rna(
        ...     model, sequence, track_indices,
        ...     gene_start=1000000, gene_end=1050000, tss=1005000
        ... )
        >>> print(f"Predicted expression: {result['mean']:.3f}")
    """
    if len(track_indices) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'n_tracks': 0,
            'track_values': np.array([])
        }
    
    # Run prediction
    pred = predict_borzoi(model, sequence, target_tracks=track_indices, device=device)
    
    # Aggregate over gene body
    aggregated = aggregate_borzoi_over_region(
        pred,
        region_start=gene_start,
        region_end=gene_end,
        seq_center=tss,
        method=aggregate_method,
        log_transform=log_transform
    )
    
    return {
        'mean': float(np.mean(aggregated)),
        'std': float(np.std(aggregated)),
        'n_tracks': len(track_indices),
        'track_values': aggregated
    }


def predict_borzoi_personal_rna(
    model: Any,
    maternal_sequence: str,
    paternal_sequence: str,
    track_indices: List[int],
    gene_start: int,
    gene_end: int,
    tss: int,
    device: str = 'cuda',
    aggregate_method: str = 'sum',
    log_transform: bool = True
) -> Dict[str, float]:
    """
    Predict RNA expression for personal genome using Borzoi.
    
    Averages predictions from maternal and paternal haplotypes,
    then aggregates over gene body using specified tracks.
    
    Args:
        model: Loaded Borzoi model
        maternal_sequence: Maternal haplotype sequence
        paternal_sequence: Paternal haplotype sequence
        track_indices: List of track indices to use for prediction
        gene_start: Gene body start (genomic coords)
        gene_end: Gene body end (genomic coords)
        tss: Transcription start site (center of input sequence)
        device: Device to run on
        aggregate_method: Aggregation method ('sum', 'mean', 'max')
        log_transform: Apply log1p transformation
        
    Returns:
        Dictionary with:
        - 'mean': Mean prediction across tracks
        - 'std': Standard deviation across tracks  
        - 'n_tracks': Number of tracks used
        - 'track_values': Array of per-track predictions
        
    Example:
        >>> # First, select tracks in notebook/script
        >>> brain_tracks = tracks_df[tracks_df['description'].str.contains('brain')]
        >>> track_indices = brain_tracks.index.tolist()
        >>> 
        >>> result = predict_borzoi_personal_rna(
        ...     model, mat_seq, pat_seq, track_indices,
        ...     gene_start=1000000, gene_end=1050000, tss=1005000
        ... )
        >>> print(f"Personal genome expression: {result['mean']:.3f}")
    """
    if len(track_indices) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'n_tracks': 0,
            'track_values': np.array([])
        }
    
    # Run personal genome prediction
    aggregated = predict_borzoi_personal_genome(
        model=model,
        maternal_sequence=maternal_sequence,
        paternal_sequence=paternal_sequence,
        target_tracks=track_indices,
        device=device,
        region_start=gene_start,
        region_end=gene_end,
        seq_center=tss,
        aggregate_method=aggregate_method,
        log_transform=log_transform
    )
    
    return {
        'mean': float(np.mean(aggregated)),
        'std': float(np.std(aggregated)),
        'n_tracks': len(track_indices),
        'track_values': aggregated
    }


# =============================================================================
# ATAC-seq / DNase Prediction (Chromatin Accessibility)
# =============================================================================

def predict_borzoi_atac(
    model: Any,
    sequence: str,
    track_indices: List[int],
    peak_start: int,
    peak_end: int,
    peak_center: int,
    device: str = 'cuda',
    aggregate_method: str = 'sum',
    log_transform: bool = True
) -> Dict[str, float]:
    """
    Predict ATAC-seq/DNase chromatin accessibility using Borzoi.
    
    Runs prediction on specified DNASE tracks and aggregates over the peak region.
    
    Args:
        model: Loaded Borzoi model
        sequence: DNA sequence (centered on peak_center, length should be BORZOI_SEQ_LEN)
        track_indices: List of DNASE track indices to use for prediction
        peak_start: Peak region start (genomic coords)
        peak_end: Peak region end (genomic coords)
        peak_center: Center position of the peak (center of input sequence)
        device: Device to run on
        aggregate_method: Aggregation method ('sum', 'mean', 'max')
        log_transform: Apply log1p transformation (recommended to match observed data)
        
    Returns:
        Dictionary with:
        - 'mean': Mean prediction across tracks
        - 'std': Standard deviation across tracks
        - 'n_tracks': Number of tracks used
        - 'track_values': Array of per-track predictions
        
    Example:
        >>> # First, find DNASE tracks for your tissue
        >>> dnase_indices = get_borzoi_dnase_track_indices(tracks_df, tissue_query='GM12878')
        >>> 
        >>> result = predict_borzoi_atac(
        ...     model, sequence, dnase_indices,
        ...     peak_start=1045000, peak_end=1046500, peak_center=1045750
        ... )
        >>> print(f"ATAC signal in peak: {result['mean']:.3f}")
    """
    if len(track_indices) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'n_tracks': 0,
            'track_values': np.array([])
        }
    
    # Run prediction
    pred = predict_borzoi(model, sequence, target_tracks=track_indices, device=device)
    
    # Aggregate over peak region
    aggregated = aggregate_borzoi_over_region(
        pred,
        region_start=peak_start,
        region_end=peak_end,
        seq_center=peak_center,
        method=aggregate_method,
        log_transform=log_transform
    )
    
    return {
        'mean': float(np.mean(aggregated)),
        'std': float(np.std(aggregated)),
        'n_tracks': len(track_indices),
        'track_values': aggregated
    }


def predict_borzoi_personal_atac(
    model: Any,
    maternal_sequence: str,
    paternal_sequence: str,
    track_indices: List[int],
    peak_start: int,
    peak_end: int,
    peak_center: int,
    device: str = 'cuda',
    aggregate_method: str = 'sum',
    log_transform: bool = True
) -> Dict[str, float]:
    """
    Predict ATAC-seq/DNase for personal genome using Borzoi.
    
    Averages predictions from maternal and paternal haplotypes,
    then aggregates over the peak region using specified DNASE tracks.
    
    This is the Borzoi equivalent of AlphaGenome's predict_personal_genome_atac().
    
    Args:
        model: Loaded Borzoi model
        maternal_sequence: Maternal haplotype sequence (centered on peak_center)
        paternal_sequence: Paternal haplotype sequence (centered on peak_center)
        track_indices: List of DNASE track indices to use for prediction
        peak_start: Peak region start (genomic coords)
        peak_end: Peak region end (genomic coords)
        peak_center: Center position of the peak (center of input sequence)
        device: Device to run on
        aggregate_method: Aggregation method ('sum', 'mean', 'max')
        log_transform: Apply log1p transformation
        
    Returns:
        Dictionary with:
        - 'mean': Mean prediction across tracks
        - 'std': Standard deviation across tracks
        - 'n_tracks': Number of tracks used
        - 'track_values': Array of per-track predictions
        
    Example:
        >>> # First, find DNASE tracks for LCL/B-cell
        >>> lcl_indices = get_borzoi_dnase_track_indices(
        ...     tracks_df, tissue_query='GM12878'
        ... )
        >>> 
        >>> result = predict_borzoi_personal_atac(
        ...     model, mat_seq, pat_seq, lcl_indices,
        ...     peak_start=1045000, peak_end=1046500, peak_center=1045750
        ... )
        >>> print(f"Personal genome ATAC: {result['mean']:.3f}")
    """
    if len(track_indices) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'n_tracks': 0,
            'track_values': np.array([])
        }
    
    # Run personal genome prediction with aggregation over peak
    aggregated = predict_borzoi_personal_genome(
        model=model,
        maternal_sequence=maternal_sequence,
        paternal_sequence=paternal_sequence,
        target_tracks=track_indices,
        device=device,
        region_start=peak_start,
        region_end=peak_end,
        seq_center=peak_center,
        aggregate_method=aggregate_method,
        log_transform=log_transform
    )
    
    return {
        'mean': float(np.mean(aggregated)),
        'std': float(np.std(aggregated)),
        'n_tracks': len(track_indices),
        'track_values': aggregated
    }


def predict_borzoi_atac_multi_tissue(
    model: Any,
    sequence: str,
    tissue_track_dict: Dict[str, List[int]],
    peak_start: int,
    peak_end: int,
    peak_center: int,
    device: str = 'cuda',
    aggregate_method: str = 'sum',
    log_transform: bool = True
) -> Dict[str, Union[float, None]]:
    """
    Predict ATAC-seq/DNase for multiple tissues using Borzoi (single API call).
    
    This is the Borzoi equivalent of AlphaGenome's predict_atac_multi_tissue().
    Unlike AlphaGenome which uses ontology terms, Borzoi requires pre-selected
    track indices per tissue.
    
    Args:
        model: Loaded Borzoi model
        sequence: DNA sequence (centered on peak_center)
        tissue_track_dict: Dictionary mapping tissue name -> list of track indices
            Example: {'brain': [1288, 1336], 'B cell': [1345, 1563]}
        peak_start: Peak region start (genomic coords)
        peak_end: Peak region end (genomic coords)
        peak_center: Center position of the peak
        device: Device to run on
        aggregate_method: Aggregation method ('sum', 'mean', 'max')
        log_transform: Apply log1p transformation
        
    Returns:
        Dictionary mapping tissue_name -> aggregated prediction value
        Returns None for tissues where prediction failed.
        Example: {'brain': 5.2, 'B cell': 4.8}
        
    Note:
        - Borzoi makes a SINGLE inference and extracts tracks for each tissue
        - More efficient than calling predict_borzoi_atac for each tissue separately
        
    Example:
        >>> # Define tissue -> track mapping
        >>> tissue_tracks = {
        ...     'brain': get_borzoi_dnase_track_indices(tracks_df, 'brain'),
        ...     'B cell': get_borzoi_dnase_track_indices(tracks_df, 'B cell'),
        ... }
        >>> 
        >>> pred_dict = predict_borzoi_atac_multi_tissue(
        ...     model, sequence, tissue_tracks,
        ...     peak_start=1045000, peak_end=1046500, peak_center=1045750
        ... )
        >>> print(f"Brain ATAC: {pred_dict['brain']:.3f}")
    """
    if not tissue_track_dict:
        return {}
    
    # Collect all unique track indices
    all_indices = []
    for indices in tissue_track_dict.values():
        all_indices.extend(indices)
    all_indices = list(set(all_indices))
    
    if len(all_indices) == 0:
        return {tissue: None for tissue in tissue_track_dict}
    
    try:
        # Single prediction with all tracks
        pred = predict_borzoi(model, sequence, target_tracks=all_indices, device=device)
        
        # Create mapping from all_indices to position in prediction array
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
        
        # Aggregate for each tissue
        tissue_predictions = {}
        for tissue, track_indices in tissue_track_dict.items():
            if len(track_indices) == 0:
                tissue_predictions[tissue] = None
                continue
                
            # Extract tracks for this tissue
            positions = [idx_to_pos[idx] for idx in track_indices if idx in idx_to_pos]
            if len(positions) == 0:
                tissue_predictions[tissue] = None
                continue
                
            tissue_pred = pred[positions, :]
            
            # Aggregate over peak region
            aggregated = aggregate_borzoi_over_region(
                tissue_pred,
                region_start=peak_start,
                region_end=peak_end,
                seq_center=peak_center,
                method=aggregate_method,
                log_transform=log_transform
            )
            
            # Average across tracks for this tissue
            tissue_predictions[tissue] = float(np.mean(aggregated))
        
        return tissue_predictions
        
    except Exception as e:
        import warnings
        warnings.warn(f"Multi-tissue ATAC prediction failed: {e}")
        return {tissue: None for tissue in tissue_track_dict}


def predict_borzoi_personal_atac_multi_tissue(
    model: Any,
    maternal_sequence: str,
    paternal_sequence: str,
    tissue_track_dict: Dict[str, List[int]],
    peak_start: int,
    peak_end: int,
    peak_center: int,
    device: str = 'cuda',
    aggregate_method: str = 'sum',
    log_transform: bool = True
) -> Dict[str, Union[float, None]]:
    """
    Predict ATAC-seq/DNase for multiple tissues on personal genome using Borzoi.
    
    This is the Borzoi equivalent of AlphaGenome's predict_personal_genome_atac_multi_tissue().
    Averages predictions from maternal and paternal haplotypes for each tissue.
    
    Args:
        model: Loaded Borzoi model
        maternal_sequence: Maternal haplotype sequence
        paternal_sequence: Paternal haplotype sequence
        tissue_track_dict: Dictionary mapping tissue name -> list of track indices
        peak_start: Peak region start (genomic coords)
        peak_end: Peak region end (genomic coords)
        peak_center: Center position of the peak
        device: Device to run on
        aggregate_method: Aggregation method ('sum', 'mean', 'max')
        log_transform: Apply log1p transformation
        
    Returns:
        Dictionary mapping tissue_name -> averaged prediction value
        Returns None for tissues where prediction failed.
        
    Note:
        - Makes TWO inference calls (maternal + paternal)
        - Each call predicts all tissues at once
        - Averages maternal and paternal predictions per tissue
        
    Example:
        >>> # Define tissue -> track mapping
        >>> tissue_tracks = {
        ...     'LCL': get_borzoi_dnase_track_indices(tracks_df, 'GM12878'),
        ...     'brain': get_borzoi_dnase_track_indices(tracks_df, 'brain'),
        ... }
        >>> 
        >>> pred_dict = predict_borzoi_personal_atac_multi_tissue(
        ...     model, mat_seq, pat_seq, tissue_tracks,
        ...     peak_start=1045000, peak_end=1046500, peak_center=1045750
        ... )
        >>> print(f"LCL ATAC: {pred_dict['LCL']:.3f}")
    """
    # Predict for both haplotypes (each returns dict of tissue -> value)
    mat_pred_dict = predict_borzoi_atac_multi_tissue(
        model, maternal_sequence, tissue_track_dict,
        peak_start, peak_end, peak_center,
        device, aggregate_method=None, log_transform=False  # Aggregate after averaging
    )
    
    pat_pred_dict = predict_borzoi_atac_multi_tissue(
        model, paternal_sequence, tissue_track_dict,
        peak_start, peak_end, peak_center,
        device, aggregate_method=None, log_transform=False
    )
    
    # Average maternal and paternal predictions per tissue
    avg_pred_dict = {}
    for tissue in tissue_track_dict:
        mat_val = mat_pred_dict.get(tissue)
        pat_val = pat_pred_dict.get(tissue)
        
        if mat_val is None or pat_val is None:
            avg_pred_dict[tissue] = None
        else:
            avg_val = (mat_val + pat_val) / 2
            # Apply aggregation and log transform
            if aggregate_method == 'sum':
                avg_val = float(np.sum(avg_val)) if isinstance(avg_val, np.ndarray) else avg_val
            elif aggregate_method == 'mean':
                avg_val = float(np.mean(avg_val)) if isinstance(avg_val, np.ndarray) else avg_val
            
            if log_transform:
                avg_val = float(np.log1p(avg_val))
                
            avg_pred_dict[tissue] = avg_val
    
    return avg_pred_dict


# =============================================================================
# Convenience Functions for Track Selection
# =============================================================================

def get_tissue_track_dict(
    tracks_df: pd.DataFrame,
    tissue_queries: List[str],
    assay_type: str = 'DNASE',
    exclude_disease: bool = True,
    exclude_embryonic: bool = True
) -> Dict[str, List[int]]:
    """
    Build a tissue -> track indices mapping for multi-tissue prediction.
    
    Convenience function to prepare tissue_track_dict for multi-tissue prediction functions.
    
    Args:
        tracks_df: DataFrame with track metadata
        tissue_queries: List of tissue query strings (e.g., ['brain', 'B cell', 'liver'])
        assay_type: Assay type prefix ('DNASE', 'RNA', 'CAGE')
        exclude_disease: Exclude disease-related samples
        exclude_embryonic: Exclude embryonic/fetal samples
        
    Returns:
        Dictionary mapping tissue query -> list of track indices
        
    Example:
        >>> tissues = ['brain', 'B cell', 'liver', 'lung']
        >>> tissue_tracks = get_tissue_track_dict(tracks_df, tissues, assay_type='DNASE')
        >>> print(f"Found {len(tissue_tracks['brain'])} brain DNASE tracks")
    """
    tissue_track_dict = {}
    
    for tissue_query in tissue_queries:
        if assay_type == 'DNASE':
            indices = get_borzoi_dnase_track_indices(
                tracks_df, tissue_query, exclude_disease, exclude_embryonic
            )
        else:
            # For other assay types, use the generic function
            indices = get_borzoi_track_indices(tracks_df, tissue_query, assay_type)
            
        tissue_track_dict[tissue_query] = indices
        
    return tissue_track_dict


# =============================================================================
# RNA Paired Strand Prediction (for ROSMAP brain expression)
# =============================================================================

def get_borzoi_rna_track_by_identifier(
    tracks_df: pd.DataFrame,
    identifier: str
) -> Tuple[int, int]:
    """
    Get +/- strand track indices for a specific RNA track identifier.
    
    RNA-seq tracks in Borzoi have paired +/- strand tracks. This function
    looks up both strand indices for a given ENCODE identifier.
    
    Args:
        tracks_df: DataFrame with track metadata (from AnnotatedBorzoi)
        identifier: ENCODE identifier (e.g., 'ENCFF196HWN')
        
    Returns:
        Tuple of (plus_strand_index, minus_strand_index)
        
    Raises:
        ValueError: If the track identifier is not found
        
    Example:
        >>> plus_idx, minus_idx = get_borzoi_rna_track_by_identifier(tracks_df, 'ENCFF196HWN')
        >>> print(f"+ strand: {plus_idx}, - strand: {minus_idx}")
        # + strand: 6407, - strand: 6408
    """
    if tracks_df is None:
        raise ValueError("tracks_df is required for track lookup")
    
    plus_track = tracks_df[tracks_df['identifier'] == f'{identifier}+']
    minus_track = tracks_df[tracks_df['identifier'] == f'{identifier}-']
    
    if len(plus_track) == 0:
        raise ValueError(f"Could not find + strand track for identifier: {identifier}")
    if len(minus_track) == 0:
        raise ValueError(f"Could not find - strand track for identifier: {identifier}")
    
    return int(plus_track.index[0]), int(minus_track.index[0])


def predict_borzoi_personal_rna_paired_strand(
    model: Any,
    maternal_sequence: str,
    paternal_sequence: str,
    plus_idx: int,
    minus_idx: int,
    gene_start: int,
    gene_end: int,
    tss: int,
    device: str = 'cuda',
    maternal_transformer: Optional["CoordinateTransformer"] = None,
    paternal_transformer: Optional["CoordinateTransformer"] = None
) -> float:
    """
    Predict RNA expression using paired +/- strand tracks for personal genome.
    
    This function implements the prediction pipeline from the ROSMAP Borzoi notebook:
    1. Predict on both haplotypes (maternal, paternal)
    2. Average the two haplotypes
    3. Sum +/- strand predictions bin-wise
    4. Aggregate over gene body (sum)
    5. Apply log1p transform
    
    This approach is used for brain RNA expression prediction with tracks like
    ENCFF196HWN (middle frontal area 46), which is similar to ROSMAP's DLPFC.
    
    Args:
        model: Loaded Borzoi/Flashzoi model
        maternal_sequence: Maternal haplotype sequence (524kb centered on TSS)
        paternal_sequence: Paternal haplotype sequence (524kb centered on TSS)
        plus_idx: Track index for + strand
        minus_idx: Track index for - strand
        gene_start: Gene body start (genomic coords)
        gene_end: Gene body end (genomic coords)
        tss: Transcription start site (center of input sequence)
        device: Device to run on ('cuda' or 'cpu')
        maternal_transformer: Optional coordinate transformer for maternal haplotype
        paternal_transformer: Optional coordinate transformer for paternal haplotype
        
    Returns:
        float: log1p(sum of combined strand signal over gene body)
        Returns np.nan if gene body doesn't overlap with prediction window.
        
    Example:
        >>> plus_idx, minus_idx = get_borzoi_rna_track_by_identifier(tracks_df, 'ENCFF196HWN')
        >>> pred = predict_borzoi_personal_rna_paired_strand(
        ...     model, mat_seq, pat_seq,
        ...     plus_idx, minus_idx,
        ...     gene_start=1000000, gene_end=1050000, tss=1005000
        ... )
        >>> print(f"Predicted expression: {pred:.3f}")
    """
    # Get raw predictions for both strands (no aggregation)
    mat_pred = predict_borzoi(model, maternal_sequence, [plus_idx, minus_idx], device)
    pat_pred = predict_borzoi(model, paternal_sequence, [plus_idx, minus_idx], device)
    
    # Sum strands bin-wise for each haplotype: shape (6144,)
    mat_combined = mat_pred[0] + mat_pred[1]
    pat_combined = pat_pred[0] + pat_pred[1]
    
    # Calculate bin indices for gene body region (per-haplotype for asymmetric indels)
    # Borzoi outputs 6144 bins covering 196,608 bp centered on input
    # Each bin is 32 bp
    half_pred = BORZOI_PRED_LENGTH // 2  # 98304
    pred_start = tss - half_pred  # Genomic start of prediction window
    
    # Transform coordinates for each haplotype if transformers provided
    mat_gene_start, mat_gene_end = gene_start, gene_end
    pat_gene_start, pat_gene_end = gene_start, gene_end
    
    if maternal_transformer is not None:
        mat_gene_start, mat_gene_end = maternal_transformer.transform_interval(gene_start, gene_end)
    if paternal_transformer is not None:
        pat_gene_start, pat_gene_end = paternal_transformer.transform_interval(gene_start, gene_end)
    
    # Calculate bin indices for each haplotype
    mat_bin_start = max(0, (mat_gene_start - pred_start) // BORZOI_BIN_SIZE)
    mat_bin_end = min(BORZOI_NUM_BINS, int(np.ceil((mat_gene_end - pred_start) / BORZOI_BIN_SIZE)))
    pat_bin_start = max(0, (pat_gene_start - pred_start) // BORZOI_BIN_SIZE)
    pat_bin_end = min(BORZOI_NUM_BINS, int(np.ceil((pat_gene_end - pred_start) / BORZOI_BIN_SIZE)))
    
    # Handle edge cases per-haplotype
    mat_valid = mat_bin_start < mat_bin_end and mat_bin_start < BORZOI_NUM_BINS and mat_bin_end > 0
    pat_valid = pat_bin_start < pat_bin_end and pat_bin_start < BORZOI_NUM_BINS and pat_bin_end > 0
    
    if not mat_valid and not pat_valid:
        return np.nan
    
    # Aggregate each haplotype separately with transformed coordinates
    mat_signal = mat_combined[mat_bin_start:mat_bin_end].sum() if mat_valid else 0.0
    pat_signal = pat_combined[pat_bin_start:pat_bin_end].sum() if pat_valid else 0.0
    
    # Average haplotypes after aggregation (handles asymmetric indels correctly)
    avg_signal = (mat_signal + pat_signal) / 2
    
    # Apply log1p transform
    return float(np.log1p(avg_signal))


# =============================================================================
# Multi-Track CSV Loading and Prediction
# =============================================================================

def load_tracks_from_csv(
    csv_path: str
) -> Tuple[List[int], Dict[str, Dict], str]:
    """
    Load track indices and metadata from a CSV file.
    
    Supports both stranded (RNA) and unstranded (ATAC/DNase) tracks.
    For RNA tracks, automatically pairs +/- strands by base identifier.
    
    Args:
        csv_path: Path to CSV file with track metadata.
                  Expected columns: index, identifier, strand, description
                  
    Returns:
        Tuple of:
        - all_indices: List of all track indices for single forward pass
        - track_info: Dict mapping base_identifier -> {
              'indices': [idx] for ATAC or [plus_idx, minus_idx] for RNA,
              'positions': [pos] or [plus_pos, minus_pos] in all_indices array,
              'description': track description,
              'strand': 'unstranded', '+', or '-'
          }
        - output_type: 'rna' or 'atac' (inferred from strand info)
        
    Example:
        >>> all_indices, track_info, output_type = load_tracks_from_csv(
        ...     'data/borzoi/selected_tracks_lcl_atac.csv'
        ... )
        >>> print(f"Loaded {len(track_info)} tracks ({output_type})")
        >>> print(f"All indices for prediction: {all_indices}")
    """
    import os
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Track CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['index', 'identifier']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV file")
    
    # Infer strand column if not present
    if 'strand' not in df.columns:
        # Infer from identifier (+ or - suffix)
        def infer_strand(identifier):
            if pd.isna(identifier):
                return 'unstranded'
            id_str = str(identifier)
            if id_str.endswith('+'):
                return '+'
            elif id_str.endswith('-'):
                return '-'
            return 'unstranded'
        df['strand'] = df['identifier'].apply(infer_strand)
    
    # Determine output type based on strand info
    has_stranded = df['strand'].isin(['+', '-']).any()
    output_type = 'rna' if has_stranded else 'atac'
    
    all_indices = []
    track_info = {}
    
    if output_type == 'atac':
        # Unstranded tracks (ATAC/DNase)
        for _, row in df.iterrows():
            idx = int(row['index'])
            identifier = str(row['identifier'])
            description = row.get('description', '')
            
            pos = len(all_indices)
            all_indices.append(idx)
            
            track_info[identifier] = {
                'indices': [idx],
                'positions': [pos],
                'description': description,
                'strand': 'unstranded'
            }
    else:
        # Stranded tracks (RNA) - pair +/- strands
        # Group by base identifier (without +/- suffix)
        strand_groups = {}
        
        for _, row in df.iterrows():
            idx = int(row['index'])
            identifier = str(row['identifier'])
            strand = str(row.get('strand', 'unstranded'))
            description = row.get('description', '')
            
            # Extract base identifier (remove +/- suffix)
            if identifier.endswith('+') or identifier.endswith('-'):
                base_id = identifier[:-1]
            else:
                base_id = identifier
            
            if base_id not in strand_groups:
                strand_groups[base_id] = {
                    'plus_idx': None,
                    'minus_idx': None,
                    'description': description
                }
            
            if strand == '+':
                strand_groups[base_id]['plus_idx'] = idx
            elif strand == '-':
                strand_groups[base_id]['minus_idx'] = idx
            else:
                # Unstranded in RNA context - treat as single track
                strand_groups[base_id]['plus_idx'] = idx
                strand_groups[base_id]['minus_idx'] = idx
        
        # Build all_indices and track_info
        for base_id, group in strand_groups.items():
            plus_idx = group['plus_idx']
            minus_idx = group['minus_idx']
            
            if plus_idx is None or minus_idx is None:
                # Skip incomplete pairs
                continue
            
            plus_pos = len(all_indices)
            all_indices.append(plus_idx)
            minus_pos = len(all_indices)
            all_indices.append(minus_idx)
            
            track_info[base_id] = {
                'indices': [plus_idx, minus_idx],
                'positions': [plus_pos, minus_pos],
                'description': group['description'],
                'strand': 'paired'
            }
    
    return all_indices, track_info, output_type


def predict_borzoi_multi_track(
    model: Any,
    maternal_sequence: str,
    paternal_sequence: str,
    all_indices: List[int],
    track_info: Dict[str, Dict],
    output_type: str,
    region_start: int,
    region_end: int,
    seq_center: int,
    device: str = 'cuda',
    maternal_transformer: Optional["CoordinateTransformer"] = None,
    paternal_transformer: Optional["CoordinateTransformer"] = None
) -> Dict[str, float]:
    """
    Run single Borzoi inference and extract per-track predictions.
    
    This function runs only 2 forward passes (maternal + paternal) regardless
    of the number of tracks, then extracts predictions for each track.
    
    For RNA tracks: sums +/- strand predictions before aggregation.
    For ATAC tracks: directly uses track predictions.
    
    Args:
        model: Loaded Borzoi/Flashzoi model
        maternal_sequence: Maternal haplotype sequence (524kb)
        paternal_sequence: Paternal haplotype sequence (524kb)
        all_indices: List of all track indices (from load_tracks_from_csv)
        track_info: Dict mapping track_id -> positions (from load_tracks_from_csv)
        output_type: 'rna' or 'atac'
        region_start: Region start (gene body or peak start)
        region_end: Region end (gene body or peak end)
        seq_center: Center position (TSS or peak center)
        device: Device to run on
        
    Returns:
        Dict mapping track_id -> prediction value (log1p transformed)
        
    Example:
        >>> all_indices, track_info, output_type = load_tracks_from_csv(csv_path)
        >>> predictions = predict_borzoi_multi_track(
        ...     model, mat_seq, pat_seq, all_indices, track_info,
        ...     output_type, gene_start, gene_end, tss
        ... )
        >>> for track_id, pred in predictions.items():
        ...     print(f"{track_id}: {pred:.3f}")
    """
    if len(all_indices) == 0:
        return {}
    
    # Run prediction on both haplotypes (2 forward passes total)
    mat_pred = predict_borzoi(model, maternal_sequence, all_indices, device)
    pat_pred = predict_borzoi(model, paternal_sequence, all_indices, device)
    
    # Calculate bin indices for region (per-haplotype for asymmetric indels)
    half_pred = BORZOI_PRED_LENGTH // 2
    pred_start = seq_center - half_pred
    
    # Transform coordinates for each haplotype if transformers provided
    mat_region_start, mat_region_end = region_start, region_end
    pat_region_start, pat_region_end = region_start, region_end
    
    if maternal_transformer is not None:
        mat_region_start, mat_region_end = maternal_transformer.transform_interval(region_start, region_end)
    if paternal_transformer is not None:
        pat_region_start, pat_region_end = paternal_transformer.transform_interval(region_start, region_end)
    
    # Calculate bin indices for each haplotype
    mat_bin_start = max(0, (mat_region_start - pred_start) // BORZOI_BIN_SIZE)
    mat_bin_end = min(BORZOI_NUM_BINS, int(np.ceil((mat_region_end - pred_start) / BORZOI_BIN_SIZE)))
    pat_bin_start = max(0, (pat_region_start - pred_start) // BORZOI_BIN_SIZE)
    pat_bin_end = min(BORZOI_NUM_BINS, int(np.ceil((pat_region_end - pred_start) / BORZOI_BIN_SIZE)))
    
    # Handle edge cases
    mat_valid = mat_bin_start < mat_bin_end and mat_bin_start < BORZOI_NUM_BINS and mat_bin_end > 0
    pat_valid = pat_bin_start < pat_bin_end and pat_bin_start < BORZOI_NUM_BINS and pat_bin_end > 0
    
    if not mat_valid and not pat_valid:
        return {track_id: np.nan for track_id in track_info}
    
    # Extract per-track predictions with per-haplotype aggregation
    predictions = {}
    
    for track_id, info in track_info.items():
        positions = info['positions']
        
        if output_type == 'rna' and len(positions) == 2:
            # RNA: sum +/- strands bin-wise, then aggregate per-haplotype
            plus_pos, minus_pos = positions
            mat_combined = mat_pred[plus_pos, :] + mat_pred[minus_pos, :]
            pat_combined = pat_pred[plus_pos, :] + pat_pred[minus_pos, :]
            
            # Aggregate each haplotype separately with transformed coordinates
            mat_signal = mat_combined[mat_bin_start:mat_bin_end].sum() if mat_valid else 0.0
            pat_signal = pat_combined[pat_bin_start:pat_bin_end].sum() if pat_valid else 0.0
            
            # Average haplotypes after aggregation
            avg_signal = (mat_signal + pat_signal) / 2
            predictions[track_id] = float(np.log1p(avg_signal))
        else:
            # ATAC: single track aggregation per-haplotype
            pos = positions[0]
            mat_signal = mat_pred[pos, mat_bin_start:mat_bin_end].sum() if mat_valid else 0.0
            pat_signal = pat_pred[pos, pat_bin_start:pat_bin_end].sum() if pat_valid else 0.0
            
            # Average haplotypes after aggregation
            avg_signal = (mat_signal + pat_signal) / 2
            predictions[track_id] = float(np.log1p(avg_signal))
    
    return predictions
