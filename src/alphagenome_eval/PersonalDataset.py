import pysam
import numpy as np
import pandas as pd

from alphagenome_eval.utils.borzoi_utils import _onehot_encoding_numba
from alphagenome_eval.utils.coordinates import CoordinateTransformer


class GenomeDataset:
    def __init__(
        self,
        gene_metadata: pd.DataFrame,
        vcf_file_path: str,
        hg38_file_path: str,
        expr_data: pd.DataFrame,
        sample_list: list[str],
        window_size: int = 20000,
        verbose: bool = False,
        std_output: bool = False,
        return_idx: bool = False,
        contig_prefix: str = "",
        genome_contig_prefix: str = "chr",
        only_personal: bool = False,
        onehot_encode: bool = False,
    ):
        """
        A dataset of gene-sample pairs, returning NumPy arrays instead of torch.Tensors.

        Args:
            gene_metadata: DataFrame with chr, tss, ensg columns
            vcf_file_path: Path to VCF file with genetic variants
            hg38_file_path: Path to reference genome FASTA file
            expr_data: DataFrame with expression values
            sample_list: List of sample IDs
            window_size: Size of genomic window around TSS (bp)
            verbose: Print debug messages
            std_output: Include standard deviation in output
            return_idx: Return gene/sample indices
            contig_prefix: Prefix for VCF contig names (e.g., '' or 'chr')
            genome_contig_prefix: Prefix for FASTA contig names (e.g., 'chr' for hg38, '' for hg19)
            only_personal: Only return personal sequences (not reference)
            onehot_encode: Return one-hot encoded sequences
        """
        self.gene_metadata = gene_metadata.reset_index(drop=True)
        self.n_genes = len(gene_metadata)
        self.vcf_file_path = vcf_file_path
        self.hg38_file_path = hg38_file_path
        self.expr_data = expr_data
        self.sample_list = list(sample_list)
        self.n_samples = len(self.sample_list)
        self.window_size = window_size
        self.verbose = verbose
        self.std_output = std_output
        self.return_idx = return_idx
        self.contig_prefix = contig_prefix
        self.genome_contig_prefix = genome_contig_prefix
        self.only_personal = only_personal
        self.onehot_encode = onehot_encode

        # Persistent file handles (opened once, reused for all queries)
        self._vcf = pysam.VariantFile(vcf_file_path, mode="r")
        self._genome = pysam.FastaFile(hg38_file_path)
        
        # Lazy variant cache: {gene_idx: {sample: [variant_dicts]}}
        # Populated on first access per gene, caches all samples at once
        self._variant_cache = {}

    def _cache_gene_variants(self, gene_idx: int, gene_info: pd.Series):
        """
        Cache variants for a gene region for ALL samples at once.
        Called lazily on first access to a gene.
        
        This eliminates redundant VCF iteration: instead of iterating through
        ~7800 VCF records for each of 180 samples (850 iterations), we iterate
        once and cache for all samples.
        """
        chr_ = gene_info["chr"]
        tss = int(gene_info["tss"])
        half = self.window_size // 2
        start = max(0, tss - half)
        end = tss + half
        
        # Initialize cache for all samples
        self._variant_cache[gene_idx] = {sample: [] for sample in self.sample_list}
        
        # Fetch VCF region ONCE
        vcf_contig = f"{self.contig_prefix}{chr_}"
        try:
            records = self._vcf.fetch(vcf_contig, start, end)
        except ValueError:
            # Contig not found in VCF
            if self.verbose:
                print(f"Warning: contig {vcf_contig} not found in VCF")
            return
        
        for rec in records:
            # Skip structural variants
            if any('<' in alt or '>' in alt for alt in rec.alts if alt):
                continue
            
            # Check ALL samples in one pass through this record
            for sample in self.sample_list:
                gt = rec.samples[sample]["GT"]
                if gt not in [(0, 0), (None, None)]:
                    # Store variant as dict for this sample
                    self._variant_cache[gene_idx][sample].append({
                        "pos": rec.pos,
                        "ref": rec.ref,
                        "alts": rec.alts,
                        "gt": gt,
                    })

    def __len__(self) -> int:
        return self.n_genes * self.n_samples

    def __getitem__(self, idx: int):
        # decode gene/sample indices
        gene_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples
        gene_info = self.gene_metadata.iloc[gene_idx]
        sample = self.sample_list[sample_idx]

        (mat_seq, pat_seq, mat_transformer, pat_transformer), ref_seq = self.process_gene_variants(gene_info, sample)

        # trim/pad to window size
        def pad_sequence(seq, target_length):
            if len(seq) >= target_length:
                return seq[:target_length]
            else:
                return seq + 'N' * (target_length - len(seq))
        
        mat_seq = pad_sequence(mat_seq, self.window_size)
        pat_seq = pad_sequence(pat_seq, self.window_size)
        ref_seq = pad_sequence(ref_seq, self.window_size)

        if self.onehot_encode:
            # one‑hot encode using numba (returns tensor of shape (4, L), convert to numpy)
            mat_enc = _onehot_encoding_numba(mat_seq, neutral_value=0.25).numpy()
            pat_enc = _onehot_encoding_numba(pat_seq, neutral_value=0.25).numpy()
            ref_enc = _onehot_encoding_numba(ref_seq, neutral_value=0.25).numpy()
        else:
            # return raw sequences as strings
            mat_enc = mat_seq
            pat_enc = pat_seq
            ref_enc = ref_seq

        # expression
        ensg = gene_info["ensg"]
        if sample in self.expr_data.columns:
            curr = self.expr_data.at[ensg, sample]
        else:
            if self.verbose:
                print(f"Warning: {sample} not in expr_data.columns")
            curr = 0.0

        mean_expr = self.expr_data.loc[ensg].mean()
        diff_expr = curr - mean_expr

        if self.std_output:
            std_expr = self.expr_data.loc[ensg].std()
            expr_arr = np.stack([mean_expr, curr, std_expr], axis=0)
            # personal sequence only
            if self.onehot_encode:
                seq_arr = np.concatenate([mat_enc, pat_enc], axis=0)
            else:
                seq_arr = np.array([mat_enc, pat_enc], dtype=object)
        else:
            if self.only_personal:
                expr_arr = np.stack([mean_expr, curr], axis=0)
            else:
                expr_arr = np.stack([mean_expr, curr], axis=0)
            # stack reference & personal sequences
            if self.onehot_encode:
                personal = np.concatenate([mat_enc, pat_enc], axis=0)
                reference = np.concatenate([ref_enc, ref_enc], axis=0)
                seq_arr = np.stack([reference, personal], axis=0)
            else:
                personal = np.array([mat_enc, pat_enc], dtype=object)
                reference = np.array([ref_enc, ref_enc], dtype=object)
                seq_arr = np.stack([reference, personal], axis=0)
        
        # Create transformer tuple
        transformers = (mat_transformer, pat_transformer)

        if self.return_idx:
            if self.onehot_encode:
                return seq_arr.astype(np.float32), expr_arr.astype(np.float32), transformers, gene_idx, sample_idx
            else:
                return seq_arr, expr_arr.astype(np.float32), transformers, gene_idx, sample_idx
        else:
            if self.onehot_encode:
                return seq_arr.astype(np.float32), expr_arr.astype(np.float32), transformers
            else:
                return seq_arr, expr_arr.astype(np.float32), transformers

    def process_gene_variants(self, gene_info: pd.Series, sample: str):
        gene_idx = gene_info.name  # DataFrame index
        chr_ = gene_info["chr"]
        tss = int(gene_info["tss"])
        half = self.window_size // 2
        start = max(0, tss - half)
        end = tss + half

        # Lazy cache: if gene not cached, process for ALL samples at once
        if gene_idx not in self._variant_cache:
            self._cache_gene_variants(gene_idx, gene_info)

        # O(1) lookup from cache
        varlist = self._variant_cache[gene_idx][sample]

        # Fetch reference sequence (fast, uses persistent handle)
        genome_contig = f"{self.genome_contig_prefix}{chr_}"
        ref_len = self._genome.get_reference_length(genome_contig)
        end = min(end, ref_len)
        seq = self._genome.fetch(genome_contig, start, end).upper()

        if self.verbose:
            print(f"Processing {len(varlist)} variants for {sample} in gene {gene_info['gene_name']} ({chr_}:{start}-{end})")

        # apply variants (using cached dict format)
        mat_seq, pat_seq, mat_transformer, pat_transformer = self.modify_sequences(seq, varlist, start)
        return (mat_seq, pat_seq, mat_transformer, pat_transformer), seq

    def modify_sequences(self, sequence, varlist, start_pos):
        """
        Apply variants to sequence and track coordinate shifts.

        Works with cached dict format:
        {"pos": int, "ref": str, "alts": tuple, "gt": tuple}

        Returns:
            mat: Maternal sequence with variants applied
            pat: Paternal sequence with variants applied
            mat_transformer: CoordinateTransformer for maternal haplotype
            pat_transformer: CoordinateTransformer for paternal haplotype
        """
        mat = pat = sequence
        mat_shift = pat_shift = 0

        # Track shift points: [(position, cumulative_shift), ...]
        mat_shifts = [(start_pos, 0)]
        pat_shifts = [(start_pos, 0)]

        for var in sorted(varlist, key=lambda v: v["pos"]):
            gt = var["gt"]
            for i, (seq_str, shift) in enumerate([(mat, mat_shift), (pat, pat_shift)]):
                # compute relative position in current shifted seq
                pos = var["pos"] - start_pos + shift - 1
                ref = var["ref"].upper()
                alt = (
                    ref if gt[i] is None or gt[i] == 0
                    else var["alts"][gt[i] - 1].upper()
                )
                # sanity check
                if seq_str[pos:pos+len(ref)] != ref:
                    if self.verbose:
                        print(f"Ref mismatch at {pos} for {'mat' if i==0 else 'pat'}: "
                              f"expected '{ref}', got '{seq_str[pos:pos+len(ref)]}'")
                    continue  # Always skip mismatched variants, regardless of verbosity
                # splice in the alt
                seq_str = seq_str[:pos] + alt + seq_str[pos+len(ref):]
                new_shift = shift + (len(alt) - len(ref))

                # Record shift point AFTER applying this variant
                if i == 0:
                    mat, mat_shift = seq_str, new_shift
                    mat_shifts.append((var["pos"], new_shift))
                else:
                    pat, pat_shift = seq_str, new_shift
                    pat_shifts.append((var["pos"], new_shift))

        # Build coordinate transformers
        mat_transformer = CoordinateTransformer(mat_shifts)
        pat_transformer = CoordinateTransformer(pat_shifts)

        return mat, pat, mat_transformer, pat_transformer
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def close(self):
        """Explicitly close file handles and release resources."""
        if hasattr(self, '_vcf') and self._vcf is not None:
            self._vcf.close()
            self._vcf = None
        if hasattr(self, '_genome') and self._genome is not None:
            self._genome.close()
            self._genome = None
        # Clear the variant cache to free memory
        self._variant_cache.clear()

    def __del__(self):
        """Destructor to ensure file handles are closed on garbage collection."""
        self.close()

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup on block exit."""
        self.close()
        return False  # Don't suppress exceptions
