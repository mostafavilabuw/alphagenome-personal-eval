#!/usr/bin/env python3
"""
Run AlphaGenome Inference Workflow

Command-line interface for running AlphaGenome predictions across
genomic regions and calculating correlations with observed data.

Usage:
    # From config file
    python scripts/run_inference.py --config configs/lcl_inference.yaml

    # Override config with CLI args
    python scripts/run_inference.py --config configs/lcl_inference.yaml --n_regions 200

    # Pure CLI mode
    python scripts/run_inference.py --data_type peaks --data_dir ./data/LCL --api_key $API_KEY ...
    
    # Multi-tissue mode with JSON file
    python scripts/run_inference.py --config configs/lcl_inference.yaml --tissue_file explore/LCL/load_data/hg_tissues.json

Author: AlphaGenome Evaluation Team
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path to import cli_utils
sys.path.insert(0, str(Path(__file__).parent))

from cli_utils import (
    load_yaml_config,
    merge_configs,
    setup_logging,
    print_config_summary,
    save_results,
    create_timestamped_dir
)

from alphagenome_eval.workflows import run_inference_workflow
from alphagenome_eval.utils import validate_borzoi_context_window, BORZOI_SUPPORTED_CONTEXT_WINDOWS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run AlphaGenome Inference Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file
  %(prog)s --config configs/lcl_inference.yaml

  # Override parameters
  %(prog)s --config configs/lcl_inference.yaml --n_regions 200 --n_samples 100

  # Use pure CLI mode
  %(prog)s --data_type peaks --data_dir /path/to/data --api_key YOUR_KEY \\
           --vcf_file_path /path/to/vcf --hg38_file_path /path/to/hg38.fa \\
           --ontology_terms EFO:0002067

Environment Variables:
  ALPHAGENOME_API_KEY    API key for AlphaGenome (can be used in config as ${ALPHAGENOME_API_KEY})
        """
    )

    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )

    # Core parameters
    parser.add_argument('--data_type', type=str, choices=['peaks', 'genes'],
                        help='Type of genomic regions (peaks or genes)')
    parser.add_argument('--api_key', type=str,
                        help='AlphaGenome API key')

    # Data paths (for peaks)
    parser.add_argument('--data_dir', type=str,
                        help='Data directory (for peaks)')

    # Data paths (for genes)
    parser.add_argument('--gene_meta_path', type=str,
                        help='Gene metadata file path (for genes)')
    parser.add_argument('--expr_path', type=str,
                        help='Expression data file path (for genes)')
    parser.add_argument('--sample_lists_path', type=str,
                        help='Sample lists directory (for genes)')

    # Common data paths
    parser.add_argument('--vcf_file_path', type=str,
                        help='VCF file path')
    parser.add_argument('--hg38_file_path', type=str,
                        help='Reference genome (hg38) file path')

    # Ontology terms
    parser.add_argument('--ontology_terms', type=str, nargs='+',
                        help='Tissue ontology terms (e.g., EFO:0002067 for LCL)')
    parser.add_argument('--tissue_file', type=str,
                        help='Path to JSON file containing tissue ontology terms (e.g., hg_tissues.json). '
                             'If provided, loads tissues from file. Takes precedence over --ontology_terms.')
    parser.add_argument('--max_tissues', type=int,
                        help='Maximum number of tissues to use from tissue_file (optional, uses all by default)')

    # Borzoi multi-track CSV support
    parser.add_argument('--borzoi_tracks_csv', type=str,
                        help='Path to CSV file with Borzoi track selections. '
                             'Enables multi-track mode with separate correlations per track. '
                             'Takes precedence over borzoi_rna_track/borzoi_atac_track.')

    # Borzoi context window (for faster VCF parsing or testing smaller contexts)
    parser.add_argument('--borzoi_context_window', type=str,
                        help='Context window size for Borzoi. Controls how much genomic '
                             'sequence to extract around TSS/peak center. Smaller windows = faster '
                             'VCF parsing but sequences are center-padded with N to 524KB. '
                             'Accepts: 4KB, 8KB, 16KB, 64KB, 128KB, 256KB, 524KB (default).')

    # Analysis parameters
    parser.add_argument('--n_regions', type=int,
                        help='Number of regions to analyze')
    parser.add_argument('--n_samples', type=int,
                        help='Number of samples per region')
    parser.add_argument('--selection_method', type=str,
                        choices=['variance', 'random', 'specific', 'predixcan'],
                        help='Region selection method (variance, random, specific, or predixcan)')
    parser.add_argument('--predixcan_results_path', type=str,
                        help='Path to PrediXcan results CSV (required for selection_method=predixcan)')
    parser.add_argument('--predixcan_metric', type=str, default='test_pearson',
                        help='PrediXcan metric to rank by (default: test_pearson, can use val_pearson). '
                             'Only applies when selection_method=predixcan.')
    parser.add_argument('--region_start_rank', type=int,
                        help='Start rank for range-based selection (1-indexed, e.g., 500 for ranks 500+). '
                             'Only applies to variance/predixcan methods. If not specified, starts from rank 1.')
    parser.add_argument('--region_end_rank', type=int,
                        help='End rank for range-based selection (1-indexed, e.g., 1000 for ranks up to 1000). '
                             'Only applies to variance/predixcan methods. If not specified, uses n_regions.')
    parser.add_argument('--window_size', type=str,
                        help='Window size around TSS/peak center. Accepts human-readable format '
                             '(2KB, 4KB, 8KB, 16KB, 100KB, 500KB, 1MB) or integer in base pairs.')
    parser.add_argument('--sequence_length', type=str,
                        choices=['2KB', '16KB', '100KB', '500KB', '1MB'],
                        help='AlphaGenome sequence length')
    parser.add_argument('--output_type', type=str, choices=['atac', 'rna'],
                        help='Output type (atac or rna)')
    parser.add_argument('--target_sample_set', type=str,
                        help='Target sample set from split file (e.g., train_samples, test_samples, val_samples)')
    parser.add_argument('--split_file_path', type=str,
                        help='Path to train_test_split.json file for using specific sample sets')
    parser.add_argument('--random_seed', type=int,
                        help='Random seed (default: 42)')

    # Parallel processing
    parser.add_argument('--enable_parallel', action='store_true',
                        help='Enable parallel processing')
    parser.add_argument('--no_parallel', dest='enable_parallel', action='store_false',
                        help='Disable parallel processing')
    parser.add_argument('--max_workers', type=int,
                        help='Maximum number of parallel workers')

    # Output
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save raw predictions to NPZ files')
    parser.add_argument('--timestamped_output', action='store_true',
                        help='Create timestamped output directory')

    # Plotting options
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--plot_per_region', action='store_true',
                        help='Create individual plots per region (can create 100s of plots, PNG only)')
    parser.add_argument('--plot_formats', type=str, nargs='+',
                        choices=['png', 'pdf'], default=['png', 'pdf'],
                        help='Plot formats for summary plots (default: png pdf)')
    parser.add_argument('--plot_dpi', type=int, default=150,
                        help='DPI for saved plots (default: 150)')

    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--log_file', type=str,
                        help='Path to log file')

    return parser.parse_args()


# Mapping from human-readable window sizes to base pairs (AlphaGenome sequence lengths)
WINDOW_SIZE_MAP = {
    '2KB': 2048,
    '4KB': 4096,
    '8KB': 8192,
    '16KB': 16384,
    '32KB': 32768,
    '64KB': 65536,
    '100KB': 102400,
    '500KB': 524288,
    '1MB': 1048576,
}

# Mapping from human-readable context window sizes to base pairs
# Uses BORZOI_SUPPORTED_CONTEXT_WINDOWS from borzoi_utils for validation
BORZOI_CONTEXT_WINDOW_MAP = {
    '4KB': 4096,
    '8KB': 8192,
    '16KB': 16384,
    '64KB': 65536,
    '100KB': 102400,
    '128KB': 131072,
    '256KB': 262144,
    '524KB': 524288,
}


def parse_window_size(value: str) -> int:
    """
    Parse a human-readable window size to base pairs.

    Args:
        value: Window size (e.g., '100KB', '500KB', '2KB') or integer string

    Returns:
        Window size in base pairs

    Raises:
        ValueError: If value is not a valid window size

    Example:
        >>> parse_window_size('100KB')
        102400
        >>> parse_window_size('500KB')
        524288
    """
    # Try human-readable format first (case-insensitive)
    value_upper = value.upper()
    if value_upper in WINDOW_SIZE_MAP:
        return WINDOW_SIZE_MAP[value_upper]

    # Try parsing as integer (for backward compatibility)
    try:
        return int(value)
    except ValueError:
        valid_str = ', '.join(WINDOW_SIZE_MAP.keys())
        raise ValueError(
            f"Invalid window size format: {value}. "
            f"Use one of: {valid_str} or an integer value in base pairs"
        )


def parse_borzoi_context_window(value: str) -> int:
    """
    Parse a human-readable context window size to base pairs.

    Args:
        value: Context window size (e.g., '16KB', '64KB', '524KB') or integer string

    Returns:
        Context window size in base pairs

    Raises:
        ValueError: If value is not a valid context window size

    Example:
        >>> parse_borzoi_context_window('16KB')
        16384
        >>> parse_borzoi_context_window('524KB')
        524288
    """
    # Try human-readable format first (case-insensitive)
    value_upper = value.upper()
    if value_upper in BORZOI_CONTEXT_WINDOW_MAP:
        return BORZOI_CONTEXT_WINDOW_MAP[value_upper]

    # Try parsing as integer (for backward compatibility)
    # Use shared validation from borzoi_utils to ensure consistency
    try:
        int_value = int(value)
        return validate_borzoi_context_window(int_value)
    except ValueError:
        valid_str = ', '.join(BORZOI_CONTEXT_WINDOW_MAP.keys())
        raise ValueError(
            f"Invalid context window format: {value}. "
            f"Use one of: {valid_str}"
        )


def load_tissues_from_json(filepath: str, max_count: Optional[int] = None) -> List[str]:
    """
    Load tissue ontology terms from JSON file.
    
    Args:
        filepath: Path to JSON file containing tissue data (e.g., hg_tissues.json)
        max_count: Maximum number of tissues to return (optional, returns all by default)
    
    Returns:
        List of tissue ontology term strings
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON file doesn't contain expected structure
    
    Example:
        >>> tissues = load_tissues_from_json('explore/LCL/load_data/hg_tissues.json', max_count=10)
        >>> print(f"Loaded {len(tissues)} tissues")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Tissue file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'ontology_terms' not in data:
            raise ValueError(
                f"Invalid tissue file format. Expected 'ontology_terms' key in JSON. "
                f"Found keys: {list(data.keys())}"
            )
        
        tissues = data['ontology_terms']
        
        if not isinstance(tissues, list):
            raise ValueError(
                f"Expected 'ontology_terms' to be a list, got {type(tissues)}"
            )
        
        if len(tissues) == 0:
            raise ValueError("No tissues found in file")
        
        # Limit to max_count if specified
        if max_count is not None and max_count > 0:
            tissues = tissues[:max_count]
        
        print(f"Loaded {len(tissues)} tissue(s) from {filepath.name}")
        if len(tissues) <= 10:
            print(f"  Tissues: {', '.join(tissues)}")
        else:
            print(f"  First 5: {', '.join(tissues[:5])}")
            print(f"  Last 5: {', '.join(tissues[-5:])}")
        
        return tissues
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Load config from file if provided
    yaml_config = None
    if args.config:
        try:
            yaml_config = load_yaml_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)

    # Merge configs (CLI args override YAML)
    cli_args = {k: v for k, v in vars(args).items() if k not in ['config', 'verbose', 'log_file', 'timestamped_output', 'no_plots', 'tissue_file', 'max_tissues']}
    config = merge_configs(yaml_config, cli_args)
    
    # Handle borzoi_context_window parsing (convert '16KB' -> 16384)
    if config.get('borzoi_context_window') is not None:
        try:
            raw_value = config['borzoi_context_window']
            # Handle both string and int (from YAML config)
            if isinstance(raw_value, str):
                config['borzoi_context_window'] = parse_borzoi_context_window(raw_value)
                print(f"✓ Borzoi context window: {raw_value} ({config['borzoi_context_window']} bp)")
            elif isinstance(raw_value, int):
                # Validate integer value
                valid_values = list(BORZOI_CONTEXT_WINDOW_MAP.values())
                if raw_value not in valid_values:
                    valid_str = ', '.join(f"{k} ({v})" for k, v in BORZOI_CONTEXT_WINDOW_MAP.items())
                    print(f"Error: Invalid borzoi_context_window: {raw_value}. Supported: {valid_str}")
                    sys.exit(1)
                print(f"✓ Borzoi context window: {raw_value} bp")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Handle window_size parsing (convert '100KB' -> 102400)
    if config.get('window_size') is not None:
        try:
            raw_value = config['window_size']
            # Handle both string and int (from YAML config)
            if isinstance(raw_value, str):
                config['window_size'] = parse_window_size(raw_value)
                print(f"✓ Window size: {raw_value} ({config['window_size']} bp)")
            elif isinstance(raw_value, int):
                # Integer value is used directly
                print(f"✓ Window size: {raw_value} bp")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Handle tissue file loading (overrides ontology_terms from config/CLI)
    # Check CLI arg first, then fall back to config file
    tissue_file = args.tissue_file or config.get('tissue_file')
    max_tissues = args.max_tissues or config.get('max_tissues')
    if tissue_file:
        try:
            tissues = load_tissues_from_json(tissue_file, max_count=max_tissues)
            config['ontology_terms'] = tissues
            print(f"✓ Loaded {len(tissues)} tissue(s) from {tissue_file}")
            if config.get('data_type') == 'peaks':
                print(f"  Multi-tissue ATAC mode enabled")
        except Exception as e:
            print(f"Error loading tissue file: {e}")
            sys.exit(1)

    # Handle plotting configuration
    config['create_plots'] = not args.no_plots
    if 'plot_per_region' not in config:
        config['plot_per_region'] = args.plot_per_region
    if 'plot_formats' not in config:
        config['plot_formats'] = args.plot_formats
    if 'plot_dpi' not in config:
        config['plot_dpi'] = args.plot_dpi

    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)

    # Create timestamped output directory if requested
    if args.timestamped_output and 'output_dir' in config:
        config['output_dir'] = str(create_timestamped_dir(config['output_dir'], prefix='inference'))

    # Print configuration
    print_config_summary(config, title="Inference Workflow Configuration")

    # Record start time
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 70)
    print(f"  AlphaGenome Inference Workflow")
    print(f"  Start Time: {start_timestamp}")
    print("=" * 70 + "\n")

    # Run workflow
    try:
        results_df, predictions = run_inference_workflow(config)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print summary
        print("\n" + "=" * 70)
        print(f"  Inference Complete!")
        print("=" * 70)
        print(f"  End Time       : {end_timestamp}")
        print(f"  Elapsed Time   : {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        print(f"  Regions        : {len(results_df)}")
        
        # Check result type: two-track RNA, multi-tissue ATAC, or single-track
        if 'pearson_corr_encode' in results_df.columns:
            # Two-track RNA results
            print(f"  Mean Pearson (Encode)  : {results_df['pearson_corr_encode'].mean():.3f}")
            print(f"  Mean Pearson (GTEx)    : {results_df['pearson_corr_gtex'].mean():.3f}")
            print(f"  Median Pearson (Encode): {results_df['pearson_corr_encode'].median():.3f}")
            print(f"  Median Pearson (GTEx)  : {results_df['pearson_corr_gtex'].median():.3f}")
        else:
            # Check for multi-tissue ATAC mode (columns like pearson_corr_EFO_0010841)
            tissue_corr_cols = [col for col in results_df.columns 
                               if col.startswith('pearson_corr_') 
                               and col not in ['pearson_corr_encode', 'pearson_corr_gtex']]
            
            if tissue_corr_cols:
                # Multi-tissue ATAC results
                all_corrs = results_df[tissue_corr_cols].values.flatten()
                all_corrs = all_corrs[~pd.isna(all_corrs)]  # Remove NaN values
                print(f"  Tissues        : {len(tissue_corr_cols)}")
                print(f"  Mean Pearson (across tissues): {all_corrs.mean():.3f}")
                print(f"  Median Pearson (across tissues): {pd.Series(all_corrs).median():.3f}")
                print(f"  Std Pearson (across tissues): {all_corrs.std():.3f}")
                # Show top 5 tissues by mean correlation
                tissue_means = {col: results_df[col].mean() for col in tissue_corr_cols}
                top_tissues = sorted(tissue_means.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top 5 tissues:")
                for col, mean_val in top_tissues:
                    tissue_name = col.replace('pearson_corr_', '').replace('_', ':')
                    print(f"    {tissue_name}: {mean_val:.3f}")
            elif 'pearson_corr' in results_df.columns:
                # Single-track results (backward compatible)
                print(f"  Mean Pearson   : {results_df['pearson_corr'].mean():.3f}")
                print(f"  Median Pearson : {results_df['pearson_corr'].median():.3f}")
                print(f"  Std Pearson    : {results_df['pearson_corr'].std():.3f}")
            else:
                print(f"  Warning: No correlation columns found in results")
        print("=" * 70 + "\n")

        # Save results
        if 'output_dir' in config:
            # Prepare metadata based on result type
            if 'pearson_corr_encode' in results_df.columns:
                # Two-track RNA results
                mean_pearson_meta = {
                    'mean_pearson_encode': float(results_df['pearson_corr_encode'].mean()),
                    'mean_pearson_gtex': float(results_df['pearson_corr_gtex'].mean())
                }
            else:
                # Check for multi-tissue ATAC mode
                tissue_corr_cols = [col for col in results_df.columns 
                                   if col.startswith('pearson_corr_') 
                                   and col not in ['pearson_corr_encode', 'pearson_corr_gtex']]
                
                if tissue_corr_cols:
                    # Multi-tissue ATAC results
                    all_corrs = results_df[tissue_corr_cols].values.flatten()
                    all_corrs = all_corrs[~np.isnan(all_corrs)]  # Remove NaN values
                    mean_pearson_meta = {
                        'mean_pearson_across_tissues': float(all_corrs.mean()),
                        'median_pearson_across_tissues': float(np.median(all_corrs)),
                        'std_pearson_across_tissues': float(all_corrs.std()),
                        'n_tissues': len(tissue_corr_cols)
                    }
                    # Add per-tissue means
                    for col in tissue_corr_cols:
                        tissue_name = col.replace('pearson_corr_', '')
                        mean_pearson_meta[f'mean_pearson_{tissue_name}'] = float(results_df[col].mean())
                elif 'pearson_corr' in results_df.columns:
                    # Single-track results (backward compatible)
                    mean_pearson_meta = {
                        'mean_pearson': float(results_df['pearson_corr'].mean())
                    }
                else:
                    # No correlation columns found
                    mean_pearson_meta = {
                        'mean_pearson': None,
                        'warning': 'No correlation columns found in results'
                    }
            
            metadata = {
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'elapsed_seconds': elapsed_time,
                'n_regions': len(results_df),
                **mean_pearson_meta,
                'config': config
            }

            results_file = save_results(
                results_df,
                output_dir=config['output_dir'],
                filename='inference_results.csv',
                metadata=metadata
            )

            print(f"Results saved to: {results_file}")
            print(f"Output directory: {config['output_dir']}")

            # Report plots if created
            if config.get('create_plots', True):
                plots_dir = Path(config['output_dir']) / 'plots'
                if plots_dir.exists():
                    print(f"Plots directory: {plots_dir}")
                    print(f"  - Summary plots: {plots_dir}")
                    if config.get('plot_per_sample', True):
                        print(f"  - Per-sample plots: {plots_dir / 'per_sample'}")
                    if config.get('plot_per_region', False):
                        print(f"  - Per-region plots: {plots_dir / 'per_region'}")
            print()

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
