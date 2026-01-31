"""
Genome sequence and variant extraction utilities.

This module provides functions for extracting genetic variants from VCF files
and generating personal genome sequences.
"""

import numpy as np
import pandas as pd
import pysam
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class VariantInfo:
    """Container for variant information."""
    chr: str
    pos: int
    ref: str
    alt: str
    maf: float
    distance_from_center: int
    variant_id: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'chr': self.chr,
            'pos': self.pos,
            'ref': self.ref,
            'alt': self.alt,
            'maf': self.maf,
            'distance_from_center': self.distance_from_center,
            'variant_id': self.variant_id
        }


def extract_variants_in_region(
    vcf_path: Union[str, Path],
    chrom: str,
    start: int,
    end: int,
    samples: List[str],
    min_maf: float = 0.05,
    contig_prefix: str = ''
) -> Tuple[np.ndarray, List[int], List[VariantInfo]]:
    """
    Extract genetic variants from VCF in a genomic region.

    This function works for both peaks and genes - it extracts all bi-allelic
    SNPs in the specified region and returns genotype dosages (0, 1, 2).

    Args:
        vcf_path: Path to VCF file
        chrom: Chromosome (e.g., '1', '2', 'X')
        start: Start position (bp)
        end: End position (bp)
        samples: List of sample IDs to extract
        min_maf: Minimum minor allele frequency threshold
        contig_prefix: Prefix for contig names in VCF ('' for no prefix, 'chr' for hg38)

    Returns:
        genotype_matrix: (n_samples, n_variants) array of dosages
        variant_positions: List of genomic positions
        variant_info: List of VariantInfo objects with detailed information

    Example:
        >>> genotypes, positions, info = extract_variants_in_region(
        ...     'data.vcf.gz', '1', 10000, 20000, ['S1', 'S2'], min_maf=0.05
        ... )
        >>> print(f"Found {len(positions)} variants")
    """
    vcf_path = Path(vcf_path)

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    genotypes = []
    variant_positions = []
    variant_infos = []

    # Calculate region center for distance calculations
    center = (start + end) // 2

    # Open VCF and fetch variants
    with pysam.VariantFile(str(vcf_path), mode="r") as vcf:
        contig = f"{contig_prefix}{chrom}"

        try:
            for record in vcf.fetch(contig, start, end):
                # Skip structural variants and multi-allelic sites
                if record.alts is None or len(record.alts) != 1:
                    continue

                # Skip complex variants (insertions/deletions marked with <> symbols)
                if any('<' in alt or '>' in alt for alt in record.alts if alt):
                    continue

                # Extract genotypes for all samples
                sample_genotypes = []
                for sample in samples:
                    try:
                        gt = record.samples[sample]["GT"]

                        # Convert GT tuple to dosage (0, 1, or 2)
                        if gt == (None, None) or gt == (0, 0):
                            dosage = 0
                        elif gt == (1, 1):
                            dosage = 2
                        elif gt in [(0, 1), (1, 0)]:
                            dosage = 1
                        else:
                            # Handle multi-allelic: count non-reference alleles
                            dosage = sum(1 for allele in gt if allele is not None and allele > 0)

                    except KeyError:
                        # Missing sample - use reference genotype
                        dosage = 0

                    sample_genotypes.append(dosage)

                # Calculate MAF
                maf = calculate_maf(np.array(sample_genotypes))

                # Filter by MAF threshold
                if maf >= min_maf:
                    genotypes.append(sample_genotypes)
                    variant_positions.append(record.pos)

                    # Store detailed variant info
                    variant_infos.append(VariantInfo(
                        chr=chrom,
                        pos=record.pos,
                        ref=record.ref,
                        alt=record.alts[0],
                        maf=maf,
                        distance_from_center=record.pos - center,
                        variant_id=f"{chrom}:{record.pos}:{record.ref}>{record.alts[0]}"
                    ))

        except Exception as e:
            # Handle cases where contig is not in VCF
            if "could not create iterator" in str(e).lower():
                # No variants in this region
                pass
            else:
                raise

    # Convert to numpy array
    if len(genotypes) == 0:
        genotype_matrix = np.array([]).reshape(len(samples), 0)
    else:
        # Transpose to (n_samples, n_variants)
        genotype_matrix = np.array(genotypes).T

    return genotype_matrix, variant_positions, variant_infos


def calculate_maf(genotype_array: np.ndarray) -> float:
    """
    Calculate minor allele frequency from genotype dosages.

    Args:
        genotype_array: Array of genotype dosages (0, 1, 2)

    Returns:
        MAF (minor allele frequency) between 0 and 0.5

    Example:
        >>> genotypes = np.array([0, 0, 1, 1, 2])
        >>> maf = calculate_maf(genotypes)
    """
    if len(genotype_array) == 0:
        return 0.0

    # Count alleles
    allele_count = np.sum(genotype_array)
    total_alleles = len(genotype_array) * 2

    if total_alleles == 0:
        return 0.0

    # Calculate frequency
    freq = allele_count / total_alleles

    # MAF is the minor allele frequency (always <= 0.5)
    maf = min(freq, 1 - freq)

    return float(maf)


def filter_variants_by_maf(
    genotype_matrix: np.ndarray,
    min_maf: float = 0.05,
    max_maf: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter genotype matrix by MAF threshold.

    Args:
        genotype_matrix: (n_samples, n_variants) array
        min_maf: Minimum MAF threshold
        max_maf: Maximum MAF threshold

    Returns:
        filtered_matrix: Filtered genotype matrix
        kept_indices: Indices of variants that passed filter

    Example:
        >>> filtered, indices = filter_variants_by_maf(genotypes, min_maf=0.05)
        >>> print(f"Kept {len(indices)} / {genotypes.shape[1]} variants")
    """
    if genotype_matrix.shape[1] == 0:
        return genotype_matrix, np.array([], dtype=int)

    # Calculate MAF for each variant (column)
    mafs = np.array([calculate_maf(genotype_matrix[:, i])
                     for i in range(genotype_matrix.shape[1])])

    # Filter by thresholds
    mask = (mafs >= min_maf) & (mafs <= max_maf)
    kept_indices = np.where(mask)[0]

    filtered_matrix = genotype_matrix[:, mask]

    return filtered_matrix, kept_indices


def get_personal_sequences(
    vcf_path: Union[str, Path],
    genome_path: Union[str, Path],
    chrom: str,
    start: int,
    end: int,
    sample_id: str,
    contig_prefix: str = ''
) -> Tuple[str, str]:
    """
    Get maternal and paternal sequences for a sample in a genomic region.

    This function extracts the personal genome sequences by applying
    variants from the VCF to the reference genome.

    Args:
        vcf_path: Path to VCF file
        genome_path: Path to reference genome FASTA
        chrom: Chromosome
        start: Start position (bp)
        end: End position (bp)
        sample_id: Sample ID to extract
        contig_prefix: Prefix for contig names

    Returns:
        maternal_sequence: DNA sequence with maternal haplotype
        paternal_sequence: DNA sequence with paternal haplotype

    Example:
        >>> mat, pat = get_personal_sequences(
        ...     'data.vcf.gz', 'hg19.fa', '1', 10000, 20000, 'HG00512'
        ... )
        >>> print(f"Sequence length: {len(mat)}")
    """
    vcf_path = Path(vcf_path)
    genome_path = Path(genome_path)

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome file not found: {genome_path}")

    # Use GenomeDataset for sequence extraction (it handles this well)
    # This is a wrapper around the existing functionality
    from alphagenome_eval import GenomeDataset

    # Create minimal metadata for this region
    region_metadata = pd.DataFrame({
        'chr': [chrom],
        'tss': [(start + end) // 2],
        'ensg': ['temp_region'],
        'gene_name': ['temp']
    })

    # Create dummy expression data
    expr_data = pd.DataFrame(
        [[0.0]],
        index=['temp_region'],
        columns=[sample_id]
    )

    # Create dataset
    dataset = GenomeDataset(
        gene_metadata=region_metadata,
        vcf_file_path=str(vcf_path),
        hg38_file_path=str(genome_path),
        expr_data=expr_data,
        sample_list=[sample_id],
        window_size=end - start,
        verbose=False,
        return_idx=False,
        contig_prefix=contig_prefix
    )

    # Extract sequences
    # dataset[0] returns (seq_arr, expr_arr) when return_idx=False
    seq_arr, expr_arr = dataset[0]
    # seq_arr has shape (2, 2) where:
    #   seq_arr[0] = reference sequences (maternal_ref, paternal_ref)
    #   seq_arr[1] = personal sequences (maternal, paternal)
    maternal_seq, paternal_seq = seq_arr[1]  # Personal sequences

    return maternal_seq, paternal_seq


def count_variants_in_regions(
    vcf_path: Union[str, Path],
    regions: pd.DataFrame,
    samples: List[str],
    min_maf: float = 0.05,
    contig_prefix: str = ''
) -> pd.DataFrame:
    """
    Count variants in multiple genomic regions.

    Args:
        vcf_path: Path to VCF file
        regions: DataFrame with chr, start, end columns
        samples: List of sample IDs
        min_maf: Minimum MAF threshold
        contig_prefix: Prefix for contig names

    Returns:
        DataFrame with variant counts per region

    Example:
        >>> counts = count_variants_in_regions(
        ...     'data.vcf.gz', peak_metadata, samples, min_maf=0.05
        ... )
        >>> print(counts[['chr', 'start', 'end', 'n_variants']].head())
    """
    results = []

    for idx, region in regions.iterrows():
        chrom = str(region['chr'])

        # Determine region boundaries
        if 'start' in region and 'end' in region:
            start = int(region['start'])
            end = int(region['end'])
        elif 'Pos_Left' in region and 'Pos_Right' in region:
            start = int(region['Pos_Left'])
            end = int(region['Pos_Right'])
        elif 'tss' in region:
            # Use TSS ± 5kb
            tss = int(region['tss'])
            start = tss - 5000
            end = tss + 5000
        else:
            raise ValueError("Region must have start/end, Pos_Left/Pos_Right, or tss")

        # Extract variants
        genotypes, positions, variant_info = extract_variants_in_region(
            vcf_path, chrom, start, end, samples, min_maf, contig_prefix
        )

        results.append({
            'region_idx': idx,
            'chr': chrom,
            'start': start,
            'end': end,
            'n_variants': len(positions),
            'mean_maf': np.mean([v.maf for v in variant_info]) if variant_info else 0.0
        })

    return pd.DataFrame(results)


def extract_variant_genotypes_for_samples(
    vcf_path: Union[str, Path],
    chrom: str,
    position: int,
    samples: List[str],
    contig_prefix: str = ''
) -> Dict[str, Tuple[int, int]]:
    """
    Extract genotypes for a specific variant across samples.

    Args:
        vcf_path: Path to VCF file
        chrom: Chromosome
        position: Genomic position
        samples: List of sample IDs
        contig_prefix: Prefix for contig names

    Returns:
        Dict mapping sample_id -> (maternal_allele, paternal_allele)

    Example:
        >>> genotypes = extract_variant_genotypes_for_samples(
        ...     'data.vcf.gz', '1', 12345, ['S1', 'S2', 'S3']
        ... )
        >>> print(genotypes['S1'])  # (0, 1) means heterozygous
    """
    vcf_path = Path(vcf_path)
    contig = f"{contig_prefix}{chrom}"

    genotypes = {}

    with pysam.VariantFile(str(vcf_path), mode="r") as vcf:
        try:
            for record in vcf.fetch(contig, position - 1, position + 1):
                if record.pos == position:
                    for sample in samples:
                        try:
                            gt = record.samples[sample]["GT"]
                            genotypes[sample] = gt if gt != (None, None) else (0, 0)
                        except KeyError:
                            genotypes[sample] = (0, 0)
                    break
        except Exception:
            # Variant not found or contig not in VCF
            for sample in samples:
                genotypes[sample] = (0, 0)

    return genotypes


def summarize_variants(variant_infos: List[VariantInfo]) -> pd.DataFrame:
    """
    Convert list of VariantInfo objects to a summary DataFrame.

    Args:
        variant_infos: List of VariantInfo objects

    Returns:
        DataFrame with variant details

    Example:
        >>> _, _, var_info = extract_variants_in_region(...)
        >>> summary_df = summarize_variants(var_info)
        >>> print(summary_df.head())
    """
    if not variant_infos:
        return pd.DataFrame(columns=['chr', 'pos', 'ref', 'alt', 'maf',
                                      'distance_from_center', 'variant_id'])

    return pd.DataFrame([v.to_dict() for v in variant_infos])