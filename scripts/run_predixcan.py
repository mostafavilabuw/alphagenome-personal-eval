#!/usr/bin/env python3
"""
Run PrediXcan Workflow

Command-line interface for running PrediXcan-style analysis:
variant extraction, ElasticNet training, and evaluation.

Usage:
    # From config file
    python scripts/run_predixcan.py --config configs/lcl_predixcan.yaml

    # Override config with CLI args
    python scripts/run_predixcan.py --config configs/lcl_predixcan.yaml --min_maf 0.01

    # Pure CLI mode
    python scripts/run_predixcan.py --data_type peaks --data_dir ./data/LCL --vcf_file_path ...

Author: AlphaGenome Evaluation Team
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

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

from alphagenome_eval.workflows import run_predixcan_workflow


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run PrediXcan Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file
  %(prog)s --config configs/lcl_predixcan.yaml

  # Override parameters
  %(prog)s --config configs/lcl_predixcan.yaml --min_maf 0.01 --context_window 200000

  # Use pure CLI mode
  %(prog)s --data_type peaks --data_dir /path/to/data \\
           --vcf_file_path /path/to/vcf --context_window 100000
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

    # Analysis parameters
    parser.add_argument('--n_regions', type=int,
                        help='Number of regions to analyze')
    parser.add_argument('--region_start_rank', type=int,
                        help='Start rank for range-based selection (1-indexed, e.g., 500 for ranks 500+). '
                             'Only applies to variance/predixcan methods. If not specified, starts from rank 1.')
    parser.add_argument('--region_end_rank', type=int,
                        help='End rank for range-based selection (1-indexed, e.g., 1000 for ranks up to 1000). '
                             'Only applies to variance/predixcan methods. If not specified, uses n_regions.')
    parser.add_argument('--context_window', type=int,
                        help='Genomic window size for variant extraction (default: 100000)')
    parser.add_argument('--min_maf', type=float,
                        help='Minimum minor allele frequency (default: 0.05)')
    parser.add_argument('--test_size', type=float,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--val_size', type=float,
                        help='Validation set fraction (default: 0.0)')
    parser.add_argument('--random_seed', type=int,
                        help='Random seed (default: 42)')

    # Model parameters
    parser.add_argument('--l1_ratio', type=float,
                        help='ElasticNet L1 ratio (default: 0.5)')
    parser.add_argument('--alphas', type=float, nargs='+',
                        help='List of alpha values to try in cross-validation')

    # Parallel processing
    parser.add_argument('--enable_parallel', type=lambda x: str(x).lower() == 'true', default=None,
                        help='Enable parallel processing (true/false)')
    parser.add_argument('--n_jobs', type=int,
                        help='Number of parallel jobs (-1 for all cores)')

    # Output
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--save_models', type=lambda x: str(x).lower() == 'true', default=None,
                        help='Save trained models to disk (true/false)')
    parser.add_argument('--save_plots', type=lambda x: str(x).lower() == 'true', default=None,
                        help='Generate and save visualization plots (true/false)')
    parser.add_argument('--timestamped_output', action='store_true',
                        help='Create timestamped output directory')

    # Predefined splits (for ROSMAP)
    parser.add_argument('--use_predefined_splits', type=lambda x: str(x).lower() == 'true', default=None,
                        help='Use predefined train/val/test splits from sample_lists_path (true/false)')

    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--log_file', type=str,
                        help='Path to log file')

    return parser.parse_args()


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

    # Merge configs (CLI args override YAML, but only if explicitly provided)
    # Filter out None values - these are unprovided CLI args that shouldn't override YAML config
    cli_args = {k: v for k, v in vars(args).items() 
                if k not in ['config', 'verbose', 'log_file', 'timestamped_output'] 
                and v is not None}
    config = merge_configs(yaml_config, cli_args)

    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)

    # Create timestamped output directory if requested
    if args.timestamped_output and 'output_dir' in config:
        config['output_dir'] = str(create_timestamped_dir(config['output_dir'], prefix='predixcan'))

    # Print configuration
    print_config_summary(config, title="PrediXcan Workflow Configuration")

    # Record start time
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 70)
    print(f"  PrediXcan Workflow")
    print(f"  Start Time: {start_timestamp}")
    print("=" * 70 + "\n")

    # Run workflow
    try:
        results_df, trained_models = run_predixcan_workflow(config)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate summary statistics
        mean_test_r2 = results_df['test_r2'].mean()
        mean_test_pearson = results_df['test_pearson'].mean()
        median_test_r2 = results_df['test_r2'].median()
        median_test_pearson = results_df['test_pearson'].median()
        mean_variants = results_df['n_variants'].mean()

        # Print summary
        print("\n" + "=" * 70)
        print(f"  PrediXcan Complete!")
        print("=" * 70)
        print(f"  End Time            : {end_timestamp}")
        print(f"  Elapsed Time        : {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        print(f"  Regions             : {len(results_df)}")
        print(f"  Mean Variants       : {mean_variants:.1f}")
        print(f"  Mean Test R²        : {mean_test_r2:.3f}")
        print(f"  Mean Test Pearson   : {mean_test_pearson:.3f}")
        print(f"  Median Test R²      : {median_test_r2:.3f}")
        print(f"  Median Test Pearson : {median_test_pearson:.3f}")

        if 'val_r2' in results_df.columns and results_df['val_r2'].notna().any():
            print(f"  Mean Val R²         : {results_df['val_r2'].mean():.3f}")
            print(f"  Mean Val Pearson    : {results_df['val_pearson'].mean():.3f}")

        print("=" * 70 + "\n")

        # Save results
        if 'output_dir' in config:
            metadata = {
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'elapsed_seconds': elapsed_time,
                'n_regions': len(results_df),
                'mean_test_r2': float(mean_test_r2),
                'mean_test_pearson': float(mean_test_pearson),
                'mean_variants': float(mean_variants),
                'n_models_trained': len(trained_models),
                'config': config
            }

            results_file = save_results(
                results_df,
                output_dir=config['output_dir'],
                filename='predixcan_results.csv',
                metadata=metadata
            )

            print(f"Results saved to: {results_file}")

            if config.get('save_models', False):
                print(f"Models saved to: {Path(config['output_dir']) / 'models'}")

            if config.get('save_plots', False):
                print(f"Plots saved to: {Path(config['output_dir']) / 'figures'}")

            print(f"Output directory: {config['output_dir']}\n")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
