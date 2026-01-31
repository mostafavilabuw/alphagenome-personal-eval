"""
Variant effect scoring utilities.

This module provides the VariantScorer class for scoring variant effects
by aggregating model predictions within genomic regions (peaks or genes).

Supports multiple aggregation methods (DIFF_SUM, LFC, etc.) and output types
(ATAC-seq, RNA-seq, etc.) with efficient prediction caching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
import warnings

try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

# Supported aggregation methods
AGGREGATION_METHODS = ['DIFF_SUM', 'DIFF_LOG2_SUM', 'DIFF_MEAN', 'LFC']

# Epsilon for log calculations to avoid log(0)
LOG_EPSILON = 1e-10


class VariantScorer:
    """
    Score variants by aggregating predictions within genomic regions.

    This class provides efficient variant effect scoring by:
    - Caching reference sequence predictions to avoid redundant API calls
    - Supporting multiple aggregation methods (DIFF_SUM, LFC, etc.)
    - Handling both ATAC-seq and RNA-seq output types

    Args:
        dna_model: AlphaGenome DNA model (from dna_client.create())
        sequence_length_bp: Sequence length in base pairs
        output_type: Type of output to predict ('ATAC', 'RNA_SEQ', 'CAGE', etc.)
        hg_fasta: Optional pysam.FastaFile for sequence extraction

    Example:
        >>> from alphagenome_eval.utils import VariantScorer, init_dna_model
        >>> model, seq_len = init_dna_model(API_KEY, '16KB')
        >>> hg_fasta = pysam.FastaFile('/path/to/hg19.fa')
        >>> scorer = VariantScorer(model, seq_len, output_type='ATAC', hg_fasta=hg_fasta)
        >>> scores = scorer.score_variant(
        ...     chrom='chr16', pos=56894519, ref='G', alt='C',
        ...     region_start=56894183, region_end=56894972,
        ...     region_center=56894577,
        ...     ontology_terms=['EFO:0010843']
        ... )
        >>> print(f"LFC score: {scores['LFC']:.4f}")
    """

    def __init__(
        self,
        dna_model,
        sequence_length_bp: int,
        output_type: str = 'ATAC',
        hg_fasta=None  # pysam.FastaFile
    ):
        self.dna_model = dna_model
        self.sequence_length_bp = sequence_length_bp
        self.output_type = output_type.upper()
        self.hg_fasta = hg_fasta
        # Cache key includes ontology_terms to avoid stale results
        self.prediction_cache: Dict[str, Dict] = {}

    def _extract_sequence(
        self,
        chrom: str,
        start: int,
        end: int
    ) -> str:
        """
        Extract DNA sequence from FASTA file.

        Args:
            chrom: Chromosome name (with or without 'chr' prefix)
            start: Start position (1-based inclusive)
            end: End position (1-based inclusive)

        Returns:
            DNA sequence string (uppercase)

        Raises:
            ValueError: If hg_fasta is not provided
        """
        if self.hg_fasta is None:
            raise ValueError(
                "hg_fasta is required for sequence extraction. "
                "Provide pysam.FastaFile in constructor."
            )

        # Ensure chr prefix
        if not chrom.startswith('chr'):
            chrom = f'chr{chrom}'

        # pysam uses 0-based half-open coordinates
        sequence = self.hg_fasta.fetch(chrom, start - 1, end).upper()
        return sequence

    def _substitute_variant(
        self,
        ref_sequence: str,
        variant_pos_in_seq: int,
        ref_allele: str,
        alt_allele: str
    ) -> str:
        """
        Substitute a variant allele into a reference sequence.

        Args:
            ref_sequence: Reference DNA sequence
            variant_pos_in_seq: 0-based position of variant in sequence
            ref_allele: Reference allele
            alt_allele: Alternate allele

        Returns:
            Alternate sequence with variant substituted

        Raises:
            ValueError: If reference allele doesn't match sequence
        """
        ref_len = len(ref_allele)
        actual_ref = ref_sequence[variant_pos_in_seq:variant_pos_in_seq + ref_len]

        if actual_ref != ref_allele:
            raise ValueError(
                f"REF allele mismatch at position {variant_pos_in_seq}: "
                f"expected '{ref_allele}', found '{actual_ref}'"
            )

        alt_sequence = (
            ref_sequence[:variant_pos_in_seq] +
            alt_allele +
            ref_sequence[variant_pos_in_seq + ref_len:]
        )
        return alt_sequence

    def _adjust_sequence_length(
        self,
        sequence: str,
        target_length: int
    ) -> str:
        """
        Adjust sequence to target length by padding with 'N' or truncating.

        Args:
            sequence: Input DNA sequence
            target_length: Desired sequence length

        Returns:
            Adjusted sequence of target length
        """
        current_length = len(sequence)
        if current_length < target_length:
            # Pad with N's at the end
            return sequence + 'N' * (target_length - current_length)
        elif current_length > target_length:
            # Truncate from end
            return sequence[:target_length]
        return sequence

    def _calculate_sequence_window(
        self,
        region_center: int
    ) -> Tuple[int, int]:
        """
        Calculate genomic coordinates for sequence window centered on region.

        Args:
            region_center: Center position of the region

        Returns:
            Tuple of (window_start, window_end) in 1-based coordinates
        """
        half_length = self.sequence_length_bp // 2
        window_start = max(1, region_center - half_length)
        window_end = window_start + self.sequence_length_bp - 1
        return window_start, window_end

    def _get_region_indices(
        self,
        region_start: int,
        region_end: int,
        region_center: int
    ) -> Tuple[int, int]:
        """
        Convert genomic region coordinates to sequence-relative array indices.

        Args:
            region_start: Genomic start position
            region_end: Genomic end position
            region_center: Center position (where sequence is centered)

        Returns:
            Tuple of (start_idx, end_idx) for array slicing
        """
        seq_center_idx = self.sequence_length_bp // 2
        start_idx = seq_center_idx + (region_start - region_center)
        end_idx = seq_center_idx + (region_end - region_center)

        # Clip to valid range
        start_idx = max(0, start_idx)
        end_idx = min(self.sequence_length_bp, end_idx)

        return start_idx, end_idx

    def _aggregate_region(
        self,
        values: np.ndarray,
        start_idx: int,
        end_idx: int,
        method: Literal['sum', 'mean'] = 'sum'
    ) -> np.ndarray:
        """
        Aggregate prediction values within a region.

        Args:
            values: Prediction array of shape (sequence_length, n_tracks)
            start_idx: Start index in array
            end_idx: End index in array
            method: 'sum' or 'mean'

        Returns:
            Aggregated values of shape (n_tracks,)
        """
        # Handle edge case of empty region
        if start_idx >= end_idx:
            warnings.warn(
                f"Empty region: start_idx ({start_idx}) >= end_idx ({end_idx}). "
                "Returning zeros."
            )
            n_tracks = values.shape[1] if values.ndim > 1 else 1
            return np.zeros(n_tracks)

        region_values = values[start_idx:end_idx, :]

        if method == 'sum':
            return region_values.sum(axis=0)
        elif method == 'mean':
            return region_values.mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _calculate_effect_scores(
        self,
        ref_sum: np.ndarray,
        alt_sum: np.ndarray,
        ref_mean: np.ndarray,
        alt_mean: np.ndarray,
        methods: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate effect scores from REF and ALT aggregated values.

        Args:
            ref_sum: Sum-aggregated REF values
            alt_sum: Sum-aggregated ALT values
            ref_mean: Mean-aggregated REF values
            alt_mean: Mean-aggregated ALT values
            methods: List of aggregation methods to compute

        Returns:
            Dictionary mapping method name to score array
        """
        scores = {}

        for method in methods:
            if method == 'DIFF_SUM':
                scores[method] = alt_sum - ref_sum
            elif method == 'DIFF_LOG2_SUM':
                scores[method] = (
                    np.log2(alt_sum + LOG_EPSILON) -
                    np.log2(ref_sum + LOG_EPSILON)
                )
            elif method == 'DIFF_MEAN':
                scores[method] = alt_mean - ref_mean
            elif method == 'LFC':
                scores[method] = np.log2(
                    (alt_mean + LOG_EPSILON) / (ref_mean + LOG_EPSILON)
                )
            else:
                warnings.warn(f"Unknown aggregation method: {method}")

        return scores

    def _get_output_values(self, prediction_output) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        Extract values, names, and metadata from prediction output based on output_type.

        Args:
            prediction_output: AlphaGenome prediction output object

        Returns:
            Tuple of (values array, track names, metadata DataFrame)
        """
        if self.output_type == 'ATAC':
            output = prediction_output.atac
        elif self.output_type == 'DNASE':
            output = prediction_output.dnase
        elif self.output_type == 'RNA_SEQ':
            output = prediction_output.rna_seq
        elif self.output_type == 'CAGE':
            output = prediction_output.cage
        else:
            raise ValueError(f"Unsupported output_type: {self.output_type}")

        return output.values, output.names, output.metadata

    def _get_output_type_enum(self):
        """Get the AlphaGenome OutputType enum for the current output_type."""
        try:
            from alphagenome.models import dna_client
        except ImportError:
            raise ImportError("AlphaGenome package not installed")

        output_type_map = {
            'ATAC': dna_client.OutputType.ATAC,
            'DNASE': dna_client.OutputType.DNASE,
            'RNA_SEQ': dna_client.OutputType.RNA_SEQ,
            'CAGE': dna_client.OutputType.CAGE,
        }

        if self.output_type not in output_type_map:
            raise ValueError(
                f"Unsupported output_type: {self.output_type}. "
                f"Must be one of {list(output_type_map.keys())}"
            )

        return output_type_map[self.output_type]

    def score_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        region_start: int,
        region_end: int,
        region_center: int,
        ontology_terms: Optional[List[str]] = None,
        aggregation_methods: Optional[List[str]] = None,
        return_track_details: bool = False
    ) -> Union[Dict[str, float], List[Dict]]:
        """
        Score a single variant within a genomic region.

        Args:
            chrom: Chromosome name
            pos: Variant position (1-based)
            ref: Reference allele
            alt: Alternate allele
            region_start: Start of region to aggregate over
            region_end: End of region to aggregate over
            region_center: Center of region (where sequence is centered)
            ontology_terms: Optional list of ontology terms for prediction
            aggregation_methods: Methods to use (default: ['DIFF_SUM', 'LFC'])
            return_track_details: If True, return per-track details

        Returns:
            If return_track_details=False:
                Dict with scores averaged across tracks:
                {'DIFF_SUM': float, 'LFC': float, 'ref_sum': float, 'alt_sum': float, ...}
            If return_track_details=True:
                List of dicts, one per track, with detailed scores

        Example:
            >>> scores = scorer.score_variant(
            ...     chrom='chr16', pos=56894519, ref='G', alt='C',
            ...     region_start=56894183, region_end=56894972,
            ...     region_center=56894577,
            ...     ontology_terms=['EFO:0010843']
            ... )
            >>> print(f"DIFF_SUM: {scores['DIFF_SUM']:.4f}")
        """
        try:
            from alphagenome.models import dna_client
        except ImportError:
            raise ImportError("AlphaGenome package not installed")

        if aggregation_methods is None:
            aggregation_methods = ['DIFF_SUM', 'LFC']

        # Normalize chromosome
        if not chrom.startswith('chr'):
            chrom = f'chr{chrom}'

        # Calculate sequence window
        window_start, window_end = self._calculate_sequence_window(region_center)

        # Cache key includes ontology_terms to avoid returning stale results
        # when different ontology terms are used for the same region
        ontology_key = ",".join(sorted(ontology_terms)) if ontology_terms else "none"
        cache_key = f"{chrom}:{window_start}-{window_end}:{ontology_key}"

        # Get or compute REF sequence prediction
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            ref_sequence = cached['sequence']
            ref_prediction = cached['output']
        else:
            # Extract reference sequence
            ref_sequence = self._extract_sequence(chrom, window_start, window_end)

            if len(ref_sequence) != self.sequence_length_bp:
                ref_sequence = self._adjust_sequence_length(
                    ref_sequence, self.sequence_length_bp
                )

            # Predict reference
            ref_prediction = self.dna_model.predict_sequence(
                sequence=ref_sequence,
                organism=dna_client.Organism.HOMO_SAPIENS,
                requested_outputs=[self._get_output_type_enum()],
                ontology_terms=ontology_terms,
            )

            # Cache
            self.prediction_cache[cache_key] = {
                'sequence': ref_sequence,
                'output': ref_prediction
            }

        # Create ALT sequence
        variant_pos_in_seq = pos - window_start

        try:
            alt_sequence = self._substitute_variant(
                ref_sequence, variant_pos_in_seq, ref, alt
            )
            if len(alt_sequence) != self.sequence_length_bp:
                alt_sequence = self._adjust_sequence_length(
                    alt_sequence, self.sequence_length_bp
                )
        except ValueError as e:
            warnings.warn(f"Variant substitution failed: {e}")
            return None

        # Predict ALT sequence
        alt_prediction = self.dna_model.predict_sequence(
            sequence=alt_sequence,
            organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=[self._get_output_type_enum()],
            ontology_terms=ontology_terms,
        )

        # Extract values
        ref_values, track_names, metadata_df = self._get_output_values(ref_prediction)
        alt_values, _, _ = self._get_output_values(alt_prediction)

        # Get region indices
        start_idx, end_idx = self._get_region_indices(
            region_start, region_end, region_center
        )

        # Aggregate predictions
        ref_sum = self._aggregate_region(ref_values, start_idx, end_idx, 'sum')
        alt_sum = self._aggregate_region(alt_values, start_idx, end_idx, 'sum')
        ref_mean = self._aggregate_region(ref_values, start_idx, end_idx, 'mean')
        alt_mean = self._aggregate_region(alt_values, start_idx, end_idx, 'mean')

        # Calculate effect scores
        effect_scores = self._calculate_effect_scores(
            ref_sum, alt_sum, ref_mean, alt_mean, aggregation_methods
        )

        if return_track_details:
            # Return per-track details
            results = []
            region_width = end_idx - start_idx

            for track_idx in range(len(track_names)):
                track_result = {
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'region_start': region_start,
                    'region_end': region_end,
                    'region_width': region_width,
                    'track_name': track_names[track_idx],
                    'ref_sum': float(ref_sum[track_idx]),
                    'alt_sum': float(alt_sum[track_idx]),
                    'ref_mean': float(ref_mean[track_idx]),
                    'alt_mean': float(alt_mean[track_idx]),
                }

                # Add metadata if available
                if track_idx < len(metadata_df):
                    track_meta = metadata_df.iloc[track_idx]
                    track_result['biosample_name'] = track_meta.get('biosample_name', '')
                    track_result['biosample_type'] = track_meta.get('biosample_type', '')
                    track_result['ontology_curie'] = track_meta.get('ontology_curie', '')

                # Add effect scores
                for method, scores in effect_scores.items():
                    track_result[method] = float(scores[track_idx])

                results.append(track_result)

            return results

        else:
            # Return averaged scores
            result = {
                'ref_sum': float(np.mean(ref_sum)),
                'alt_sum': float(np.mean(alt_sum)),
                'ref_mean': float(np.mean(ref_mean)),
                'alt_mean': float(np.mean(alt_mean)),
            }

            for method, scores in effect_scores.items():
                result[method] = float(np.mean(scores))

            return result

    def score_variants_batch(
        self,
        variants_df: pd.DataFrame,
        regions_df: Optional[pd.DataFrame] = None,
        chrom_col: str = 'CHROM',
        pos_col: str = 'POS',
        ref_col: str = 'REF',
        alt_col: str = 'ALT',
        region_id_col: str = 'region_id',
        region_start_col: str = 'region_start',
        region_end_col: str = 'region_end',
        region_center_col: str = 'region_center',
        ontology_terms: Optional[List[str]] = None,
        aggregation_methods: Optional[List[str]] = None,
        return_track_details: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Score multiple variants efficiently with caching.

        Args:
            variants_df: DataFrame with variant information
            regions_df: Optional DataFrame with region information (if not in variants_df)
            chrom_col: Column name for chromosome
            pos_col: Column name for position
            ref_col: Column name for reference allele
            alt_col: Column name for alternate allele
            region_id_col: Column to join variants_df with regions_df
            region_start_col: Column for region start
            region_end_col: Column for region end
            region_center_col: Column for region center
            ontology_terms: Ontology terms for prediction
            aggregation_methods: Methods to compute
            return_track_details: If True, return per-track results
            show_progress: Show progress bar

        Returns:
            DataFrame with scoring results

        Example:
            >>> results_df = scorer.score_variants_batch(
            ...     variants_df,
            ...     ontology_terms=['EFO:0010843'],
            ...     aggregation_methods=['DIFF_SUM', 'LFC']
            ... )
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x  # fallback if tqdm not installed

        if aggregation_methods is None:
            aggregation_methods = ['DIFF_SUM', 'LFC']

        # Merge region info if provided separately
        if regions_df is not None:
            df = variants_df.merge(regions_df, on=region_id_col, how='left')
        else:
            df = variants_df.copy()

        all_results = []

        iterator = df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(df), desc="Scoring variants")

        for idx, row in iterator:
            try:
                result = self.score_variant(
                    chrom=str(row[chrom_col]),
                    pos=int(row[pos_col]),
                    ref=str(row[ref_col]),
                    alt=str(row[alt_col]),
                    region_start=int(row[region_start_col]),
                    region_end=int(row[region_end_col]),
                    region_center=int(row[region_center_col]),
                    ontology_terms=ontology_terms,
                    aggregation_methods=aggregation_methods,
                    return_track_details=return_track_details
                )

                if result is None:
                    continue

                if return_track_details:
                    # Add variant identifiers to each track result
                    for track_result in result:
                        track_result['variant_idx'] = idx
                        all_results.append(track_result)
                else:
                    result['variant_idx'] = idx
                    result[chrom_col] = row[chrom_col]
                    result[pos_col] = row[pos_col]
                    result[ref_col] = row[ref_col]
                    result[alt_col] = row[alt_col]
                    all_results.append(result)

            except Exception as e:
                warnings.warn(f"Failed to score variant at index {idx}: {e}")
                continue

        if not all_results:
            return pd.DataFrame()

        return pd.DataFrame(all_results)

    def clear_cache(self):
        """Clear the prediction cache."""
        self.prediction_cache = {}

    @property
    def cache_size(self) -> int:
        """Return the number of cached predictions."""
        return len(self.prediction_cache)
