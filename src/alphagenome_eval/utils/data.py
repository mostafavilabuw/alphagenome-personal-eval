"""
Data loading utilities for AlphaGenome evaluation.

This module provides simple functions for loading peak and gene data,
creating genome datasets, and managing train/test splits.
"""

import pandas as pd
import numpy as np
import pysam
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union


def load_peaks(
    data_dir: Union[str, Path],
    n_peaks: Optional[int] = None,
    selection_method: str = 'variance',
    specific_indices: Optional[List[int]] = None,
    predixcan_results_path: Optional[Union[str, Path]] = None,
    predixcan_metric: str = 'test_pearson',
    filter_chromosomes: Optional[List[str]] = None,
    random_state: int = 42,
    start_rank: Optional[int] = None,
    end_rank: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load LCL ATAC-seq peak data.

    Args:
        data_dir: Directory containing peaks.bed.gz and log_fpkm.tsv.gz
        n_peaks: Number of peaks to select (None = all peaks)
        selection_method: Selection method:
            - 'variance': Peaks with highest variance (default)
            - 'random': Random peaks
            - 'specific': Use specific_indices list
            - 'predixcan': Top N peaks by PrediXcan correlation (requires predixcan_results_path)
        specific_indices: Indices to use if selection_method='specific'
        predixcan_results_path: Path to PrediXcan results CSV (required for selection_method='predixcan')
        predixcan_metric: Column name to rank by when selection_method='predixcan' (default: 'test_pearson', can use 'val_pearson')
        filter_chromosomes: List of chromosomes to keep (default: 1-22, X, Y)
        random_state: Random seed for reproducibility
        start_rank: Start rank for range selection (1-indexed, e.g., 500). Only for variance/predixcan.
        end_rank: End rank for range selection (1-indexed, e.g., 1000). Only for variance/predixcan.

    Returns:
        peak_metadata: DataFrame with chr, Pos_Left, Pos_Right, tss, ensg, gene_name
        expression_data: DataFrame indexed by ensg (log-transformed FPKM)
        sample_names: List of sample IDs

    Example:
        >>> # Select top 1000 by variance
        >>> metadata, expr, samples = load_peaks('./data/LCL', n_peaks=1000)
        >>> print(f"Loaded {len(metadata)} peaks × {len(samples)} samples")
        
        >>> # Select ranks 500-1000 by variance
        >>> metadata, expr, samples = load_peaks('./data/LCL', start_rank=500, end_rank=1000)
        
        >>> # Select by PrediXcan performance
        >>> metadata, expr, samples = load_peaks(
        ...     './data/LCL', 
        ...     n_peaks=50,
        ...     selection_method='predixcan',
        ...     predixcan_results_path='./results/predixcan_results.csv'
        ... )
    """
    data_dir = Path(data_dir)
    peaks_file = data_dir / "peaks.bed.gz"
    fpkm_file = data_dir / "log_fpkm.tsv.gz"

    # Validate files exist
    if not peaks_file.exists():
        raise FileNotFoundError(f"Peaks file not found: {peaks_file}")
    if not fpkm_file.exists():
        raise FileNotFoundError(f"Expression file not found: {fpkm_file}")

    # Load raw data
    peaks_df = pd.read_csv(peaks_file, sep='\t', compression='gzip')
    log_fpkm_df = pd.read_csv(fpkm_file, sep='\t', compression='gzip', index_col=0)
    sample_names = log_fpkm_df.columns.tolist()

    # Validate data alignment
    if len(peaks_df) != len(log_fpkm_df):
        raise ValueError(
            f"Data mismatch: {len(peaks_df)} peaks in metadata but "
            f"{len(log_fpkm_df)} peaks in expression data"
        )

    # Align indices
    log_fpkm_df.index = peaks_df.index

    # Standardize metadata fields
    peaks_df = peaks_df.copy()
    peaks_df['tss'] = (peaks_df['Pos_Left'] + peaks_df['Pos_Right']) // 2
    peaks_df['chr'] = peaks_df['#Chr'].astype(str).str.replace('.0', '', regex=False)
    peaks_df['ensg'] = peaks_df.index.astype(str)
    peaks_df['gene_name'] = 'Peak_' + peaks_df.index.astype(str)

    # Filter to standard chromosomes
    if filter_chromosomes is None:
        filter_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
    chr_mask = peaks_df['chr'].isin(filter_chromosomes)
    peaks_df = peaks_df[chr_mask].copy()
    log_fpkm_df = log_fpkm_df[chr_mask].copy()

    # Validate we have data after filtering
    if len(peaks_df) == 0:
        raise ValueError("No peaks found after chromosome filtering")

    # Validate rank parameters if provided
    if start_rank is not None or end_rank is not None:
        if start_rank is not None and start_rank < 1:
            raise ValueError(f"start_rank must be >= 1, got {start_rank}")
        if end_rank is not None and end_rank < 1:
            raise ValueError(f"end_rank must be >= 1, got {end_rank}")
        if start_rank is not None and end_rank is not None and start_rank >= end_rank:
            raise ValueError(f"start_rank ({start_rank}) must be < end_rank ({end_rank})")

    # Select peaks if requested
    if n_peaks is not None or start_rank is not None or end_rank is not None:
        n_available = len(peaks_df)
        
        # Determine selection mode: range-based or top N
        use_range_selection = (start_rank is not None) or (end_rank is not None)
        
        if use_range_selection:
            # Range-based selection (e.g., ranks 500-1000)
            actual_start = start_rank if start_rank is not None else 1
            actual_end = end_rank if end_rank is not None else n_available
            
            # Validate range
            if actual_end > n_available:
                warnings.warn(
                    f"Requested end_rank {actual_end} exceeds available peaks {n_available}. "
                    f"Using {n_available} as end_rank."
                )
                actual_end = n_available
            
            # Convert to 0-indexed for slicing
            slice_start = actual_start - 1
            slice_end = actual_end
        else:
            # Top N selection (existing behavior)
            n_peaks_to_select = min(n_peaks, n_available) if n_peaks is not None else n_available

            if n_peaks and n_peaks > n_available:
                warnings.warn(
                    f"Requested {n_peaks} peaks but only {n_available} available. "
                    f"Using all {n_available} peaks."
                )

        if selection_method == 'variance':
            # Select peaks by variance ranking
            variance = log_fpkm_df.var(axis=1)
            sorted_by_variance = variance.sort_values(ascending=False)
            
            if use_range_selection:
                top_indices = sorted_by_variance.iloc[slice_start:slice_end].index
            else:
                top_indices = sorted_by_variance.iloc[:n_peaks_to_select].index
                
        elif selection_method == 'random':
            # Random selection (not affected by rank parameters)
            if use_range_selection:
                warnings.warn("Range selection (start_rank/end_rank) not applicable to random selection method")
            np.random.seed(random_state)
            n_to_select = n_peaks_to_select if not use_range_selection else (slice_end - slice_start)
            top_indices = np.random.choice(peaks_df.index, n_to_select, replace=False)
            
        elif selection_method == 'specific':
            if specific_indices is None:
                raise ValueError("Must provide specific_indices when selection_method='specific'")
            if use_range_selection:
                warnings.warn("Range selection (start_rank/end_rank) not applicable to specific selection method")
            top_indices = specific_indices
        elif selection_method == 'predixcan':
            # Select peaks by PrediXcan correlation ranking (test_pearson or val_pearson)
            if predixcan_results_path is None:
                raise ValueError("Must provide predixcan_results_path when selection_method='predixcan'")
            
            # Load PrediXcan results (don't filter by val_pearson to allow flexibility)
            predixcan_results = load_predixcan_results(predixcan_results_path, filter_valid=False)
            
            # Sort by specified metric descending
            if predixcan_metric not in predixcan_results.columns:
                available_cols = ', '.join(predixcan_results.columns.tolist())
                raise ValueError(
                    f"PrediXcan results must contain '{predixcan_metric}' column. "
                    f"Available columns: {available_cols}"
                )
            
            sorted_predixcan = predixcan_results.sort_values(predixcan_metric, ascending=False)
            
            # Apply range or top N selection
            if use_range_selection:
                selected_predixcan = sorted_predixcan.iloc[slice_start:slice_end]
            else:
                selected_predixcan = sorted_predixcan.iloc[:n_peaks_to_select]
            
            # Extract region indices - try multiple possible column names
            if 'region_id' in selected_predixcan.columns:
                top_region_ids = selected_predixcan['region_id'].astype(int).values
            elif 'peak_idx' in selected_predixcan.columns:
                top_region_ids = selected_predixcan['peak_idx'].astype(int).values
            else:
                # Use index if no explicit ID column
                top_region_ids = selected_predixcan.index.values
            
            # Filter to indices that exist in our data
            available_indices = peaks_df.index.tolist()
            top_indices = [idx for idx in top_region_ids if idx in available_indices]
            
            if len(top_indices) == 0:
                raise ValueError(
                    f"No matching peaks found between PrediXcan results and peak data. "
                    f"PrediXcan region IDs: {top_region_ids[:5]}..., "
                    f"Available peak indices: {available_indices[:5]}..."
                )
            
            expected_count = slice_end - slice_start if use_range_selection else n_peaks_to_select
            if len(top_indices) < expected_count:
                warnings.warn(
                    f"Only {len(top_indices)} of requested {expected_count} peaks "
                    f"found in PrediXcan results"
                )
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")

        peaks_df = peaks_df.loc[top_indices]
        log_fpkm_df = log_fpkm_df.loc[top_indices]

    # Reset index to sequential integers
    peaks_df = peaks_df.reset_index(drop=True)

    # Prepare expression data with ensg identifiers
    log_fpkm_df.index = peaks_df['ensg'].values

    return peaks_df, log_fpkm_df, sample_names


def load_genes(
    gene_meta_path: Union[str, Path],
    expr_data_path: Union[str, Path],
    sample_lists_path: Optional[Union[str, Path]] = None,
    filter_chromosomes: Optional[List[str]] = None,
    select_genes: Optional[List[str]] = None,
    n_genes: Optional[int] = None,
    selection_method: str = 'variance',
    predixcan_results_path: Optional[Union[str, Path]] = None,
    predixcan_metric: str = 'test_pearson',
    random_state: int = 42,
    start_rank: Optional[int] = None,
    end_rank: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Load ROSMAP gene expression data with flexible gene selection.

    Args:
        gene_meta_path: Path to gene metadata TSV file
        expr_data_path: Path to expression data CSV file
        sample_lists_path: Path to directory containing sample list CSVs
        filter_chromosomes: List of chromosomes to keep (default: 1-22, X, Y)
        select_genes: List of ENSG IDs to select (for selection_method='specific')
        n_genes: Number of genes to select (None = all genes)
        selection_method: Method for gene selection:
            - 'variance': Genes with highest expression variance (default, best for ML)
            - 'random': Random genes
            - 'specific': Use select_genes list (requires select_genes parameter)
            - 'predixcan': Top N genes by PrediXcan correlation (requires predixcan_results_path)
        predixcan_results_path: Path to PrediXcan results CSV (required for selection_method='predixcan')
        predixcan_metric: Column name to rank by when selection_method='predixcan' (default: 'test_pearson', can use 'val_pearson')
        random_state: Random seed for reproducibility
        start_rank: Start rank for range selection (1-indexed, e.g., 500). Only for variance/predixcan.
        end_rank: End rank for range selection (1-indexed, e.g., 1000). Only for variance/predixcan.

    Returns:
        gene_metadata: DataFrame with chr, tss, start, end, ensg, gene_name
        expression_data: DataFrame indexed by ensg
        sample_lists: Dict of {'all_subs': [...], 'train_subs': [...],
                               'val_subs': [...], 'test_subs': [...]}

    Examples:
        >>> # Select top 1000 by variance (default)
        >>> metadata, expr, splits = load_genes(
        ...     'gene-ids-and-positions.tsv',
        ...     'expression_data.csv',
        ...     n_genes=100,
        ...     selection_method='variance'
        ... )
        
        >>> # Select random genes
        >>> metadata, expr, splits = load_genes(
        ...     'gene-ids-and-positions.tsv',
        ...     'expression_data.csv',
        ...     n_genes=100,
        ...     selection_method='random'
        ... )
        
        >>> # Select specific genes
        >>> metadata, expr, splits = load_genes(
        ...     'gene-ids-and-positions.tsv',
        ...     'expression_data.csv',
        ...     select_genes=['ENSG00000139618', 'ENSG00000141510'],
        ...     selection_method='specific'
        ... )
        
        >>> # Select by PrediXcan performance
        >>> metadata, expr, splits = load_genes(
        ...     'gene-ids-and-positions.tsv',
        ...     'expression_data.csv',
        ...     n_genes=50,
        ...     selection_method='predixcan',
        ...     predixcan_results_path='./results/predixcan_results.csv'
        ... )
    """
    gene_meta_path = Path(gene_meta_path)
    expr_data_path = Path(expr_data_path)

    # Load gene metadata
    gene_meta = pd.read_csv(gene_meta_path, sep="\t")

    # Standardize column names
    gene_meta = gene_meta.copy()
    gene_meta['tss'] = gene_meta['tss_hg38']
    gene_meta['chr'] = gene_meta['chr_hg38'].astype(str)
    gene_meta['ensg'] = gene_meta['gene_id']

    # Add start/end if not present (use tss as fallback)
    if 'start_hg38' in gene_meta.columns:
        gene_meta['start'] = gene_meta['start_hg38']
    elif 'start' not in gene_meta.columns:
        gene_meta['start'] = gene_meta['tss']

    if 'end_hg38' in gene_meta.columns:
        gene_meta['end'] = gene_meta['end_hg38']
    elif 'end' not in gene_meta.columns:
        gene_meta['end'] = gene_meta['tss']

    # Filter chromosomes
    if filter_chromosomes is None:
        filter_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
    gene_meta = gene_meta[gene_meta['chr'].isin(filter_chromosomes)].copy()

    # Load expression data
    expr_data = pd.read_csv(expr_data_path, index_col=0)

    # Validate rank parameters if provided
    if start_rank is not None or end_rank is not None:
        if start_rank is not None and start_rank < 1:
            raise ValueError(f"start_rank must be >= 1, got {start_rank}")
        if end_rank is not None and end_rank < 1:
            raise ValueError(f"end_rank must be >= 1, got {end_rank}")
        if start_rank is not None and end_rank is not None and start_rank >= end_rank:
            raise ValueError(f"start_rank ({start_rank}) must be < end_rank ({end_rank})")

    # Apply gene selection based on method
    if selection_method == 'specific':
        # Use provided gene list
        if select_genes is None:
            raise ValueError("selection_method='specific' requires select_genes parameter")
        if start_rank is not None or end_rank is not None:
            warnings.warn("Range selection (start_rank/end_rank) not applicable to specific selection method")
        gene_meta = gene_meta[gene_meta['ensg'].isin(select_genes)].copy()
        # Only keep genes that exist in expression data
        valid_genes = [g for g in select_genes if g in expr_data.index]
        expr_data = expr_data.loc[valid_genes]
    
    elif n_genes is not None or start_rank is not None or end_rank is not None:
        # Select genes based on selection_method
        # Ensure genes exist in both metadata and expression data
        common_genes = list(set(gene_meta['ensg'].values) & set(expr_data.index))
        
        # Determine selection mode: range-based or top N
        use_range_selection = (start_rank is not None) or (end_rank is not None)
        
        if use_range_selection:
            # Range-based selection (e.g., ranks 500-1000)
            actual_start = start_rank if start_rank is not None else 1
            actual_end = end_rank if end_rank is not None else len(common_genes)
            
            # Validate range
            if actual_end > len(common_genes):
                warnings.warn(
                    f"Requested end_rank {actual_end} exceeds available genes {len(common_genes)}. "
                    f"Using {len(common_genes)} as end_rank."
                )
                actual_end = len(common_genes)
            
            # Convert to 0-indexed for slicing
            slice_start = actual_start - 1
            slice_end = actual_end
        
        if selection_method == 'variance':
            # Select genes by variance ranking
            gene_variances = expr_data.loc[common_genes].var(axis=1)
            sorted_by_variance = gene_variances.sort_values(ascending=False)
            
            if use_range_selection:
                top_genes = sorted_by_variance.iloc[slice_start:slice_end].index.tolist()
            else:
                n_to_select = min(n_genes, len(common_genes)) if n_genes is not None else len(common_genes)
                top_genes = sorted_by_variance.iloc[:n_to_select].index.tolist()
            
        elif selection_method == 'random':
            # Random selection (not affected by rank parameters)
            if use_range_selection:
                warnings.warn("Range selection (start_rank/end_rank) not applicable to random selection method")
                n_to_select = slice_end - slice_start
            else:
                n_to_select = min(n_genes, len(common_genes)) if n_genes is not None else len(common_genes)
            np.random.seed(random_state)
            top_genes = np.random.choice(common_genes, size=n_to_select, replace=False).tolist()
        
        elif selection_method == 'predixcan':
            # Select genes by PrediXcan correlation ranking (test_pearson or val_pearson)
            if predixcan_results_path is None:
                raise ValueError("Must provide predixcan_results_path when selection_method='predixcan'")
            
            # Load PrediXcan results (don't filter by val_pearson to allow flexibility)
            predixcan_results = load_predixcan_results(predixcan_results_path, filter_valid=False)
            
            # Sort by specified metric descending
            if predixcan_metric not in predixcan_results.columns:
                available_cols = ', '.join(predixcan_results.columns.tolist())
                raise ValueError(
                    f"PrediXcan results must contain '{predixcan_metric}' column. "
                    f"Available columns: {available_cols}"
                )
            
            sorted_predixcan = predixcan_results.sort_values(predixcan_metric, ascending=False)
            
            # Apply range or top N selection
            if use_range_selection:
                selected_predixcan = sorted_predixcan.iloc[slice_start:slice_end]
            else:
                n_to_select = min(n_genes, len(sorted_predixcan)) if n_genes is not None else len(sorted_predixcan)
                selected_predixcan = sorted_predixcan.iloc[:n_to_select]
            
            # Extract gene IDs (ENSG) - try multiple possible column names
            if 'region_id' in selected_predixcan.columns:
                top_gene_ids = selected_predixcan['region_id'].tolist()
            elif 'ensg' in selected_predixcan.columns:
                top_gene_ids = selected_predixcan['ensg'].tolist()
            elif 'region_name' in selected_predixcan.columns:
                # Some formats use region_name for ENSG IDs
                top_gene_ids = selected_predixcan['region_name'].tolist()
            else:
                # Use index if no explicit ID column
                top_gene_ids = selected_predixcan.index.tolist()
            
            # Filter to genes that exist in both metadata and expression data
            top_genes = [g for g in top_gene_ids if g in common_genes]
            
            if len(top_genes) == 0:
                raise ValueError(
                    f"No matching genes found between PrediXcan results and gene data. "
                    f"PrediXcan gene IDs: {top_gene_ids[:5]}..., "
                    f"Available genes: {common_genes[:5]}..."
                )
            
            expected_count = len(selected_predixcan)
            if len(top_genes) < expected_count:
                warnings.warn(
                    f"Only {len(top_genes)} of requested {expected_count} genes "
                    f"found in PrediXcan results"
                )
            
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}. "
                           f"Use 'variance', 'random', 'specific', or 'predixcan'")
        
        # Filter metadata and expression data
        gene_meta = gene_meta[gene_meta['ensg'].isin(top_genes)].copy()
        expr_data = expr_data.loc[top_genes]

    # Load sample lists if path provided
    sample_lists = {}
    if sample_lists_path is not None:
        sample_lists_path = Path(sample_lists_path)

        for split in ['all_subs', 'train_subs', 'val_subs', 'test_subs']:
            file_path = sample_lists_path / f'{split}.csv'

            if file_path.exists():
                raw_subs = pd.read_csv(file_path, header=None, dtype=str)[0].tolist()
                # ROSMAP-specific format: P12345_P12345
                formatted_subs = [f'P{i}_P{i}' for i in raw_subs]
                sample_lists[split] = formatted_subs
            else:
                warnings.warn(f"Sample list file not found: {file_path}")

    return gene_meta, expr_data, sample_lists


def load_vcf_samples(vcf_path: Union[str, Path]) -> List[str]:
    """
    Get sample names from VCF file header.

    Args:
        vcf_path: Path to VCF file

    Returns:
        List of sample IDs

    Example:
        >>> samples = load_vcf_samples('data.vcf.gz')
        >>> print(f"VCF contains {len(samples)} samples")
    """
    vcf_path = Path(vcf_path)

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    with pysam.VariantFile(str(vcf_path), mode="r") as vcf:
        samples = list(vcf.header.samples)

    return samples


def load_predixcan_results(
    results_path: Union[str, Path],
    filter_valid: bool = True,
    min_pearson: Optional[float] = None
) -> pd.DataFrame:
    """
    Load PrediXcan results from CSV file.

    Args:
        results_path: Path to PrediXcan results CSV
        filter_valid: Remove rows with missing performance metrics
        min_pearson: Minimum validation Pearson correlation to keep (None = no filter)

    Returns:
        DataFrame with PrediXcan results

    Example:
        >>> results = load_predixcan_results('predixcan_results.csv', min_pearson=0.1)
        >>> print(f"Found {len(results)} genes with val_pearson >= 0.1")
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"PrediXcan results file not found: {results_path}")

    results = pd.read_csv(results_path)

    # Filter to valid results if requested
    if filter_valid:
        # Remove rows with missing performance metrics
        if 'val_pearson' in results.columns:
            results = results.dropna(subset=['val_pearson'])
        elif 'test_pearson' in results.columns:
            results = results.dropna(subset=['test_pearson'])

    # Filter by minimum Pearson correlation
    if min_pearson is not None:
        if 'val_pearson' in results.columns:
            results = results[results['val_pearson'] >= min_pearson]
        elif 'test_pearson' in results.columns:
            results = results[results['test_pearson'] >= min_pearson]
        else:
            warnings.warn("No pearson column found, min_pearson filter not applied")

    return results


def create_genome_dataset(
    metadata: pd.DataFrame,
    vcf_file_path: Union[str, Path],
    genome_file_path: Union[str, Path],
    expression_data: pd.DataFrame,
    sample_list: List[str],
    window_size: int,
    verbose: bool = False,
    random_state: int = 42,
    return_idx: bool = False,
    contig_prefix: str = '',
    genome_contig_prefix: str = 'chr',
    onehot_encode: bool = False
) -> 'GenomeDataset':
    """
    Create GenomeDataset from metadata (works for both peaks and genes).

    This is a convenience wrapper around the GenomeDataset class that
    automatically validates inputs and handles common configuration.

    Args:
        metadata: DataFrame with chr, tss, ensg columns (peaks or genes)
        vcf_file_path: Path to VCF file with genetic variants
        genome_file_path: Path to reference genome FASTA file
        expression_data: DataFrame with expression/accessibility values
        sample_list: List of sample IDs to include
        window_size: Size of genomic window around TSS (bp)
        verbose: Print progress messages
        return_idx: Return sample index in dataset
        contig_prefix: Prefix for contig names ('' for hg19, 'chr' for hg38)
        onehot_encode: Return one-hot encoded sequences

    Returns:
        GenomeDataset instance

    Example:
        >>> dataset = create_genome_dataset(
        ...     metadata=peak_metadata,
        ...     vcf_file_path='genotypes.vcf.gz',
        ...     genome_file_path='hg19.fa',
        ...     expression_data=log_fpkm,
        ...     sample_list=samples,
        ...     window_size=100000
        ... )
        >>> seq, expr, _, idx = dataset[0]
    """
    from alphagenome_eval import GenomeDataset

    # Validate required columns
    required_cols = ['chr', 'tss', 'ensg']
    missing = [c for c in required_cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    # Validate sample list first (quick check before expensive file checks)
    if len(sample_list) == 0:
        raise ValueError("sample_list cannot be empty")

    # Validate files exist
    vcf_path = Path(vcf_file_path)
    genome_path = Path(genome_file_path)

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome file not found: {genome_path}")

    # Create dataset
    dataset = GenomeDataset(
        gene_metadata=metadata,
        vcf_file_path=str(vcf_path),
        hg38_file_path=str(genome_path),
        expr_data=expression_data,
        sample_list=sample_list,
        window_size=window_size,
        verbose=verbose,
        return_idx=return_idx,
        contig_prefix=contig_prefix,
        genome_contig_prefix=genome_contig_prefix,
        onehot_encode=onehot_encode
    )

    return dataset


def align_samples(
    expression_data: pd.DataFrame,
    vcf_samples: List[str],
    sample_lists: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[str]]:
    """
    Align sample names between expression data, VCF, and sample lists.

    This helper function finds the intersection of samples across different
    data sources and optionally validates against provided sample lists.

    Args:
        expression_data: DataFrame with samples as columns
        vcf_samples: List of sample IDs from VCF
        sample_lists: Optional dict of sample lists (e.g., train/val/test splits)

    Returns:
        Dict with aligned samples: {'available': [...], 'train': [...], ...}

    Example:
        >>> aligned = align_samples(expr_data, vcf_samples, splits)
        >>> print(f"Found {len(aligned['available'])} samples in common")
    """
    # Get samples from expression data
    expr_samples = set(expression_data.columns)
    vcf_samples_set = set(vcf_samples)

    # Find intersection
    available_samples = list(expr_samples.intersection(vcf_samples_set))

    result = {'available': available_samples}

    # Validate against sample lists if provided
    if sample_lists is not None:
        for split_name, split_samples in sample_lists.items():
            # Find samples that are both in the split and available
            valid_samples = [s for s in split_samples if s in available_samples]
            result[split_name] = valid_samples

            # Warn if some samples are missing
            missing = len(split_samples) - len(valid_samples)
            if missing > 0:
                warnings.warn(
                    f"{missing} samples in '{split_name}' not found in data"
                )

    return result