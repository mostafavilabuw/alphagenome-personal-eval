"""
CLI Utilities for AlphaGenome Evaluation Scripts

Shared utilities for command-line interface scripts, including
YAML config loading, logging setup, and result saving.

Author: AlphaGenome Evaluation Team
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def load_yaml_config(config_path: str) -> Dict:
    """
    Load YAML configuration file with environment variable expansion.

    Supports ${ENV_VAR} or $ENV_VAR syntax for environment variables.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read and expand environment variables
    with open(config_path, 'r') as f:
        content = f.read()

    # Expand environment variables
    content = os.path.expandvars(content)

    # Parse YAML
    config = yaml.safe_load(content)

    return config


def merge_configs(yaml_config: Optional[Dict], cli_args: Dict) -> Dict:
    """
    Merge YAML config with command-line arguments.

    CLI arguments override YAML values. Only non-None CLI args are used.

    Args:
        yaml_config: Configuration from YAML file (or None)
        cli_args: Arguments from command line (argparse namespace as dict)

    Returns:
        Merged configuration dictionary
    """
    # Start with YAML config or empty dict
    merged = yaml_config.copy() if yaml_config else {}

    # Override with non-None CLI arguments
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value

    return merged


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for CLI scripts.

    Args:
        verbose: If True, set level to DEBUG, otherwise INFO
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def print_config_summary(config: Dict, title: str = "Configuration") -> None:
    """
    Pretty print configuration dictionary.

    Args:
        config: Configuration dictionary to print
        title: Title for the summary
    """
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    for key, value in sorted(config.items()):
        # Format value
        if isinstance(value, (list, tuple)):
            if len(value) > 3:
                value_str = f"[{value[0]}, {value[1]}, ... ({len(value)} items)]"
            else:
                value_str = str(value)
        elif isinstance(value, dict):
            value_str = f"{{...}} ({len(value)} keys)"
        elif isinstance(value, str) and len(value) > 50:
            value_str = value[:47] + "..."
        else:
            value_str = str(value)

        print(f"  {key:25s} : {value_str}")

    print("=" * 70 + "\n")


def save_results(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = "results.csv",
    metadata: Optional[Dict] = None
) -> Path:
    """
    Save results DataFrame with metadata.

    Creates output directory if it doesn't exist and saves results as CSV.
    Optionally saves metadata as JSON.

    Args:
        results_df: Results DataFrame to save
        output_dir: Output directory path
        filename: Output filename (default: results.csv)
        metadata: Optional metadata dictionary to save as JSON

    Returns:
        Path to saved results file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results
    results_file = output_path / filename
    results_df.to_csv(results_file, index=False)

    # Save metadata if provided
    if metadata:
        import json
        metadata_file = output_path / filename.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    return results_file


def create_timestamped_dir(base_dir: str, prefix: str = "") -> Path:
    """
    Create a timestamped output directory.

    Args:
        base_dir: Base directory path
        prefix: Optional prefix for directory name

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}" if prefix else timestamp

    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def validate_required_keys(config: Dict, required_keys: list, config_name: str = "config") -> None:
    """
    Validate that required keys are present in configuration.

    Args:
        config: Configuration dictionary
        required_keys: List of required key names
        config_name: Name of configuration (for error messages)

    Raises:
        ValueError: If any required key is missing
    """
    missing = [key for key in required_keys if key not in config]

    if missing:
        raise ValueError(
            f"Missing required {config_name} keys: {', '.join(missing)}\n"
            f"Please provide these either in config file or via command-line arguments."
        )
