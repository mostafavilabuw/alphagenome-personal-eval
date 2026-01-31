"""
Visualization utilities for AlphaGenome evaluation.

This module provides reusable plotting functions for:
- Prediction vs observed scatter plots
- Correlation heatmaps
- Distribution plots
- Performance comparison plots
- Genomic track visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any, Union
from scipy.stats import pearsonr, spearmanr
from pathlib import Path


# ============================================================================
# Anti-Artifact Helper Functions
# ============================================================================

def _prepare_figure_for_save(fig) -> None:
    """
    Prepare figure for artifact-free saving.

    Addresses common rendering issues:
    - Vector tiling artifacts (1-pixel seams)
    - Transparency layering artifacts
    - Grid line bleeding

    Args:
        fig: Matplotlib figure object
    """
    # Set solid white background on figure
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    # Set solid white background on all axes
    for ax in fig.get_axes():
        ax.set_facecolor('white')
        ax.patch.set_alpha(1.0)
        ax.grid(False)  # Explicitly disable grid


def _save_figure_clean(fig, save_path: Union[str, Path], dpi: int = 150) -> None:
    """
    Save figure with anti-artifact settings.

    Args:
        fig: Matplotlib figure object
        save_path: Path to save figure
        dpi: Resolution for raster formats
    """
    _prepare_figure_for_save(fig)
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        transparent=False
    )


# ============================================================================
# Helper Functions for Format Detection
# ============================================================================

def _detect_result_format(results_df: pd.DataFrame) -> str:
    """
    Detect if results are single-track (ATAC), two-track (RNA), multi-tissue ATAC,
    or multi-track CSV format.
    
    Args:
        results_df: Results DataFrame
        
    Returns:
        'single' for single-tissue ATAC (has 'pearson_corr'), 
        'two_track' for RNA (has 'pearson_corr_encode' and 'pearson_corr_gtex'),
        'multi_tissue' for multi-tissue ATAC (has multiple 'pearson_corr_<tissue>' columns with EFO/CL/UBERON prefixes),
        'multi_track' for multi-track CSV mode (has multiple 'pearson_corr_ENCFF*' columns)
    """
    # Check for two-track RNA format
    if 'pearson_corr_encode' in results_df.columns and 'pearson_corr_gtex' in results_df.columns:
        return 'two_track'
    
    # Check for multi-tissue ATAC format (multiple pearson_corr_<tissue> columns)
    # Look for columns matching pattern pearson_corr_<tissue> where tissue starts with EFO/CL/UBERON
    corr_cols = [col for col in results_df.columns if col.startswith('pearson_corr_')]
    tissue_corr_cols = [col for col in corr_cols 
                        if any(col.startswith(f'pearson_corr_{prefix}') 
                               for prefix in ['EFO', 'CL', 'UBERON'])]
    
    if len(tissue_corr_cols) > 1:
        return 'multi_tissue'
    
    # Check for multi-track CSV format (multiple pearson_corr_ENCFF* columns)
    track_corr_cols = [col for col in corr_cols 
                       if col.startswith('pearson_corr_ENCFF')]
    
    if len(track_corr_cols) >= 1:
        return 'multi_track'
    
    # Default to single-tissue ATAC
    return 'single'


def _get_predictions_array(region_data: Dict, track: str = 'encode', tissue: str = None) -> np.ndarray:
    """
    Extract predictions array from region_data, handling multiple formats.
    
    Args:
        region_data: Dictionary with prediction data
        track: For two-track RNA data, which track to use ('encode' or 'gtex').
               For multi-track mode, use track identifier (e.g., 'ENCFF915DFR').
        tissue: For multi-tissue ATAC data, which tissue to use (e.g., 'EFO:0010841').
                If None and multi-tissue detected, uses first tissue in tissue_list.
        
    Returns:
        Predictions array
    """
    # Check if this is two-track format (RNA)
    if 'predictions_encode' in region_data:
        if track == 'gtex':
            return region_data['predictions_gtex']
        else:  # default to encode
            return region_data['predictions_encode']
    
    # Check if this is multi-track CSV format
    # In the predictions_dict from workflow, predictions are nested under 'predictions_per_track'
    elif 'predictions_per_track' in region_data:
        # Get track to use
        track_list = region_data.get('track_list', [])
        predictions_per_track = region_data['predictions_per_track']
        
        if track and track in predictions_per_track:
            return np.array(predictions_per_track[track])
        elif track_list:
            # Use first track as default
            first_track = track_list[0]
            return np.array(predictions_per_track[first_track])
        elif predictions_per_track:
            # Fallback: use first key from predictions_per_track
            first_track = list(predictions_per_track.keys())[0]
            return np.array(predictions_per_track[first_track])
        else:
            raise KeyError(f"Track '{track}' not found in predictions_per_track. "
                          f"Available tracks: {list(predictions_per_track.keys())}")
    
    # Check if this is multi-tissue format (ATAC with multiple tissues)
    # In the predictions_dict from workflow, predictions are nested under 'predictions_per_tissue'
    elif 'predictions_per_tissue' in region_data:
        # Get tissue to use
        if tissue is None:
            # Use first tissue as default
            tissue = region_data['tissue_list'][0]
        
        # Access predictions from the nested dictionary
        predictions_per_tissue = region_data['predictions_per_tissue']
        
        if tissue in predictions_per_tissue:
            return np.array(predictions_per_tissue[tissue])
        else:
            raise KeyError(f"Tissue '{tissue}' not found in predictions_per_tissue. "
                          f"Available tissues: {list(predictions_per_tissue.keys())}")
    
    # Check if this is multi-tissue format loaded from NPZ file
    # (Used when loading saved predictions, keys are like 'predictions_EFO_0010841')
    elif 'tissue_list' in region_data:
        # Get tissue to use
        if tissue is None:
            # Use first tissue as default
            tissue = region_data['tissue_list'][0]
        
        # Convert tissue format if needed (EFO:0010841 -> EFO_0010841)
        tissue_key = tissue.replace(':', '_')
        pred_key = f'predictions_{tissue_key}'
        
        if pred_key in region_data:
            return region_data[pred_key]
        else:
            raise KeyError(f"Tissue '{tissue}' (key: '{pred_key}') not found in predictions. "
                          f"Available tissues: {list(region_data['tissue_list'])}")
    
    else:
        # Single-track format (ATAC)
        return region_data['predictions']


def _match_predixcan_to_results(
    results_df: pd.DataFrame,
    predixcan_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match PrediXcan results to inference results by region.

    Tries matching by region_id first, then falls back to (chr, start, end) matching.

    Args:
        results_df: Inference results DataFrame with region_id or (chr, start, end)
        predixcan_df: PrediXcan results DataFrame with region_id or (chr, start, end)

    Returns:
        Tuple of (matched_results_df, matched_predixcan_df) with aligned rows
    """
    # Try matching by region_id first
    if 'region_id' in results_df.columns and 'region_id' in predixcan_df.columns:
        # Direct merge on region_id
        common_ids = set(results_df['region_id']) & set(predixcan_df['region_id'])
        if len(common_ids) > 0:
            matched_results = results_df[results_df['region_id'].isin(common_ids)].copy()
            matched_predixcan = predixcan_df[predixcan_df['region_id'].isin(common_ids)].copy()

            # Ensure same order
            matched_results = matched_results.set_index('region_id').loc[list(common_ids)].reset_index()
            matched_predixcan = matched_predixcan.set_index('region_id').loc[list(common_ids)].reset_index()

            return matched_results, matched_predixcan

    # Try matching by coordinates (chr, start, end)
    coord_cols = ['chr', 'start', 'end']
    if all(col in results_df.columns for col in coord_cols) and all(col in predixcan_df.columns for col in coord_cols):
        # Create coordinate key for matching
        results_df = results_df.copy()
        predixcan_df = predixcan_df.copy()

        results_df['_coord_key'] = results_df['chr'].astype(str) + ':' + results_df['start'].astype(str) + '-' + results_df['end'].astype(str)
        predixcan_df['_coord_key'] = predixcan_df['chr'].astype(str) + ':' + predixcan_df['start'].astype(str) + '-' + predixcan_df['end'].astype(str)

        common_coords = set(results_df['_coord_key']) & set(predixcan_df['_coord_key'])
        if len(common_coords) > 0:
            matched_results = results_df[results_df['_coord_key'].isin(common_coords)].copy()
            matched_predixcan = predixcan_df[predixcan_df['_coord_key'].isin(common_coords)].copy()

            # Ensure same order
            matched_results = matched_results.set_index('_coord_key').loc[list(common_coords)].reset_index()
            matched_predixcan = matched_predixcan.set_index('_coord_key').loc[list(common_coords)].reset_index()

            # Clean up temporary column
            matched_results = matched_results.drop(columns=['_coord_key'])
            matched_predixcan = matched_predixcan.drop(columns=['_coord_key'])

            return matched_results, matched_predixcan

    # No matching possible
    return pd.DataFrame(), pd.DataFrame()


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_prediction_vs_observed(
    y_observed: Union[np.ndarray, pd.Series],
    y_predicted: Union[np.ndarray, pd.Series],
    title: str = 'Prediction vs Observed',
    xlabel: str = 'Observed',
    ylabel: str = 'Predicted',
    show_correlation: bool = True,
    show_identity: bool = False,
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    color: str = 'steelblue',
    alpha: float = 0.6
) -> plt.Figure:
    """
    Create scatter plot comparing predictions to observed values.

    Args:
        y_observed: Observed values
        y_predicted: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_correlation: Display Pearson and Spearman correlations
        show_identity: Show y=x identity line
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        dpi: DPI for saved figure
        color: Point color
        alpha: Point transparency

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_prediction_vs_observed(
        ...     y_test, test_pred,
        ...     title='Test Set Performance',
        ...     save_path='results/test_scatter.png'
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy arrays
    if isinstance(y_observed, pd.Series):
        y_observed = y_observed.values
    if isinstance(y_predicted, pd.Series):
        y_predicted = y_predicted.values

    # Remove NaN values
    mask = ~(np.isnan(y_observed) | np.isnan(y_predicted))
    y_obs_clean = y_observed[mask]
    y_pred_clean = y_predicted[mask]

    # Scatter plot
    ax.scatter(y_obs_clean, y_pred_clean, c=color, alpha=alpha, edgecolors='k', linewidth=0.5)

    # Identity line
    if show_identity:
        lim_min = min(y_obs_clean.min(), y_pred_clean.min())
        lim_max = max(y_obs_clean.max(), y_pred_clean.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1.5, label='y=x')

    # Calculate and display correlations
    if show_correlation and len(y_obs_clean) > 1:
        pearson_r, pearson_p = pearsonr(y_obs_clean, y_pred_clean)
        spearman_r, spearman_p = spearmanr(y_obs_clean, y_pred_clean)

        # Add text box with statistics
        textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})\nn = {len(y_obs_clean)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if show_identity:
        ax.legend()

    plt.tight_layout()

    # Save if requested - with anti-artifact settings
    if save_path is not None:
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('white')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)

    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    method: str = 'pearson',
    title: str = 'Correlation Heatmap',
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'coolwarm',
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create correlation heatmap for multiple variables.

    Args:
        data: DataFrame with samples as rows, variables as columns
        method: 'pearson' or 'spearman'
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        annot: Show correlation values in cells
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> corr_df = pd.DataFrame({
        ...     'observed': obs_values,
        ...     'predicted': pred_values,
        ...     'predixcan': predixcan_values
        ... })
        >>> fig = plot_correlation_heatmap(
        ...     corr_df, title='Model Comparison', save_path='heatmap.png'
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = data.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = data.corr(method='spearman')
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'pearson' or 'spearman'")

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        _save_figure_clean(fig, save_path, dpi)

    return fig


def plot_distribution(
    data: Union[np.ndarray, pd.Series, List[float]],
    title: str = 'Distribution',
    xlabel: str = 'Value',
    ylabel: str = 'Frequency',
    bins: int = 30,
    show_stats: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'skyblue',
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create histogram with optional statistics overlay.

    Args:
        data: Data to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of histogram bins
        show_stats: Show mean and median lines
        figsize: Figure size
        color: Histogram color
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_distribution(
        ...     test_correlations,
        ...     title='Test Pearson Correlations',
        ...     xlabel='Pearson r',
        ...     save_path='correlation_dist.png'
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy array
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data)

    # Remove NaN values
    data_clean = data[~np.isnan(data)]

    # Histogram - use matching edgecolor to prevent vector tiling artifacts
    ax.hist(data_clean, bins=bins, alpha=0.7, color=color, edgecolor=color, linewidth=0.5)

    # Statistics
    if show_stats and len(data_clean) > 0:
        mean_val = np.mean(data_clean)
        median_val = np.median(data_clean)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.3f}')

        ax.legend()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)  # Disable grid to prevent rendering artifacts

    plt.tight_layout()

    # Save with anti-artifact settings
    if save_path is not None:
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('white')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)

    return fig


def plot_correlation_distributions(
    correlation_data: Dict[str, Union[np.ndarray, pd.Series]],
    titles: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (20, 4),
    bins: int = 15,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot multiple correlation distributions side-by-side.

    Args:
        correlation_data: Dict mapping names to correlation arrays
        titles: Optional dict mapping names to plot titles
        figsize: Figure size
        bins: Number of histogram bins
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> data = {
        ...     'encode_vs_label': encode_corrs,
        ...     'gtex_vs_label': gtex_corrs,
        ...     'predixcan_test': predixcan_corrs
        ... }
        >>> fig = plot_correlation_distributions(data, save_path='dists.png')
    """
    n_plots = len(correlation_data)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]

    for i, (name, data) in enumerate(correlation_data.items()):
        ax = axes[i]

        # Convert to numpy array and remove NaN
        if isinstance(data, pd.Series):
            data = data.values
        data_clean = data[~np.isnan(data)]

        # Plot histogram - use linewidth=0.5 with matching edgecolor to prevent artifacts
        ax.hist(data_clean, bins=bins, alpha=0.7, edgecolor='white', linewidth=0.5)

        # Add statistics
        if len(data_clean) > 0:
            mean_val = np.mean(data_clean)
            median_val = np.median(data_clean)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7,
                       label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', alpha=0.7,
                       label=f'Median: {median_val:.3f}')
            ax.legend(fontsize=8)

        # Labels
        title = titles.get(name, name) if titles else name
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Pearson r', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(False)  # Disable grid to prevent rendering artifacts

    plt.tight_layout()

    # Save with anti-artifact settings
    if save_path is not None:
        fig.patch.set_facecolor('white')
        for ax in axes if isinstance(axes, list) else [axes]:
            ax.set_facecolor('white')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)

    return fig


def plot_train_vs_test_comparison(
    train_metric: Union[np.ndarray, pd.Series],
    test_metric: Union[np.ndarray, pd.Series],
    metric_name: str = 'R²',
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create scatter plot comparing training and test metrics.

    Args:
        train_metric: Training metric values
        test_metric: Test metric values
        metric_name: Name of metric for labels
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_train_vs_test_comparison(
        ...     train_r2, test_r2, metric_name='R²', save_path='train_test.png'
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy arrays
    if isinstance(train_metric, pd.Series):
        train_metric = train_metric.values
    if isinstance(test_metric, pd.Series):
        test_metric = test_metric.values

    # Remove NaN values
    mask = ~(np.isnan(train_metric) | np.isnan(test_metric))
    train_clean = train_metric[mask]
    test_clean = test_metric[mask]

    # Scatter plot
    ax.scatter(train_clean, test_clean, alpha=0.6, color='coral', edgecolors='k', linewidth=0.5)

    # Identity line
    lim_min = min(train_clean.min(), test_clean.min())
    lim_max = max(train_clean.max(), test_clean.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1.5, label='y=x')

    # Calculate correlation
    if len(train_clean) > 1:
        corr_r, _ = pearsonr(train_clean, test_clean)
        ax.text(0.05, 0.95, f'Pearson r = {corr_r:.3f}\nn = {len(train_clean)}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(f'Training {metric_name}', fontsize=12)
    ax.set_ylabel(f'Test {metric_name}', fontsize=12)
    ax.set_title(f'Training vs Test {metric_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(False)  # Disable grid to prevent rendering artifacts

    plt.tight_layout()

    if save_path is not None:
        _save_figure_clean(fig, save_path, dpi)

    return fig


def plot_performance_vs_variants(
    n_variants: Union[np.ndarray, pd.Series],
    performance_metric: Union[np.ndarray, pd.Series],
    metric_name: str = 'Test R²',
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot performance metric vs number of variants.

    Args:
        n_variants: Number of variants per region
        performance_metric: Performance metric values
        metric_name: Name of performance metric
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_performance_vs_variants(
        ...     results_df['n_variants'],
        ...     results_df['test_r2'],
        ...     save_path='performance_vs_variants.png'
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy arrays
    if isinstance(n_variants, pd.Series):
        n_variants = n_variants.values
    if isinstance(performance_metric, pd.Series):
        performance_metric = performance_metric.values

    # Remove NaN values
    mask = ~(np.isnan(n_variants) | np.isnan(performance_metric))
    variants_clean = n_variants[mask]
    metric_clean = performance_metric[mask]

    # Scatter plot
    ax.scatter(variants_clean, metric_clean, alpha=0.6, color='mediumpurple',
               edgecolors='k', linewidth=0.5)

    # Calculate correlation
    if len(variants_clean) > 1:
        corr_r, corr_p = pearsonr(variants_clean, metric_clean)
        ax.text(0.05, 0.95, f'Pearson r = {corr_r:.3f}\np = {corr_p:.2e}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Number of Variants', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} vs Variant Count', fontsize=14, fontweight='bold')
    ax.grid(False)  # Disable grid to prevent rendering artifacts

    plt.tight_layout()

    if save_path is not None:
        _save_figure_clean(fig, save_path, dpi)

    return fig


def plot_boxplot_comparison(
    data_dict: Dict[str, Union[np.ndarray, pd.Series]],
    title: str = 'Comparison',
    ylabel: str = 'Value',
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create box plot comparing multiple groups.

    Args:
        data_dict: Dict mapping group names to data arrays
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save figure
        dpi: DPI for saved figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> data = {
        ...     'Training': train_r2,
        ...     'Validation': val_r2,
        ...     'Test': test_r2
        ... }
        >>> fig = plot_boxplot_comparison(data, ylabel='R²', save_path='boxplot.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    labels = list(data_dict.keys())
    data_arrays = []

    for name, data in data_dict.items():
        if isinstance(data, pd.Series):
            data = data.values
        # Remove NaN
        data_clean = data[~np.isnan(data)]
        data_arrays.append(data_clean)

    # Box plot
    bp = ax.boxplot(data_arrays, labels=labels, patch_artist=True,
                    notch=True, showmeans=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_arrays)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)  # Disable grid to prevent rendering artifacts

    plt.tight_layout()

    if save_path is not None:
        _save_figure_clean(fig, save_path, dpi)

    return fig


def create_comprehensive_predixcan_plots(
    results_df: pd.DataFrame,
    output_dir: Union[str, Path],
    prefix: str = 'predixcan',
    dpi: int = 150
) -> Dict[str, Path]:
    """
    Create comprehensive set of PrediXcan analysis plots.

    Args:
        results_df: DataFrame with PrediXcan results
        output_dir: Directory to save plots
        prefix: Prefix for saved files
        dpi: DPI for saved figures

    Returns:
        Dictionary mapping plot names to saved file paths

    Example:
        >>> plot_paths = create_comprehensive_predixcan_plots(
        ...     results_df, output_dir='./figures', prefix='lcl_predixcan'
        ... )
        >>> print(f"Created {len(plot_paths)} plots")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_plots = {}

    # Filter to successful results (if status column exists)
    if 'status' in results_df.columns:
        # Filter for successful results (success or unknown status)
        successful = results_df[results_df['status'].isin(['success', 'unknown'])].copy()
    else:
        # No status column - use all results
        successful = results_df.copy()

    if len(successful) == 0:
        print("Warning: No results to plot")
        return saved_plots

    # 1. Test correlation distribution
    if 'test_pearson' in successful.columns:
        path = output_dir / f'{prefix}_test_correlation_dist.png'
        plot_distribution(
            successful['test_pearson'],
            title='Test Pearson Correlation Distribution',
            xlabel='Pearson r',
            save_path=path,
            dpi=dpi
        )
        saved_plots['test_correlation_dist'] = path
        plt.close()

    # 2. Train vs test comparison
    if 'train_pearson' in successful.columns and 'test_pearson' in successful.columns:
        path = output_dir / f'{prefix}_train_vs_test.png'
        plot_train_vs_test_comparison(
            successful['train_pearson'],
            successful['test_pearson'],
            metric_name='Pearson r',
            save_path=path,
            dpi=dpi
        )
        saved_plots['train_vs_test'] = path
        plt.close()

    # 3. Performance vs variants
    if 'n_variants' in successful.columns and 'test_pearson' in successful.columns:
        path = output_dir / f'{prefix}_performance_vs_variants.png'
        plot_performance_vs_variants(
            successful['n_variants'],
            successful['test_pearson'],
            metric_name='Test Pearson r',
            save_path=path,
            dpi=dpi
        )
        saved_plots['performance_vs_variants'] = path
        plt.close()

    return saved_plots


def create_inference_plots(
    results_df: pd.DataFrame,
    predictions_dict: Dict[str, Dict],
    output_dir: Union[str, Path],
    plot_per_sample: bool = True,
    plot_per_region: bool = False,
    plot_formats: List[str] = ['png', 'pdf'],
    dpi: int = 150,
    predixcan_results_path: Optional[Union[str, Path]] = None,
    model_name: str = 'Model'
) -> Dict[str, List[Path]]:
    """
    Create comprehensive inference plots from workflow results.

    Args:
        results_df: DataFrame with columns [region_id, region_name, chr, n_samples,
                    pearson_corr, p_value, mean_pred, std_pred, mean_obs, std_obs]
        predictions_dict: Dict mapping region_id -> {
            'predictions': np.array,
            'observed': np.array,
            'sample_ids': list
        }
        output_dir: Directory to save plots
        plot_per_sample: Create individual plots per sample (default: True)
        plot_per_region: Create individual plots per region (default: False, can be many)
        plot_formats: List of formats to save ['png', 'pdf'], or subset
        dpi: DPI for saved figures
        predixcan_results_path: Optional path to PrediXcan results CSV for comparison
        model_name: Name of the model for labeling (e.g., 'AlphaGenome', 'Borzoi')

    Returns:
        Dictionary mapping plot types to lists of saved file paths

    Example:
        >>> plot_paths = create_inference_plots(
        ...     results_df, predictions,
        ...     output_dir='./results/plots',
        ...     plot_per_region=True,
        ...     predixcan_results_path='./results/predixcan_results.csv'
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_plots = {
        'summary': [],
        'per_sample': [],
        'per_region': []
    }

    # Load PrediXcan results if path provided
    predixcan_df = None
    if predixcan_results_path:
        predixcan_path = Path(predixcan_results_path)
        if predixcan_path.exists():
            try:
                predixcan_df = pd.read_csv(predixcan_path)
                if 'test_pearson' not in predixcan_df.columns:
                    print(f"Warning: PrediXcan file missing 'test_pearson' column: {predixcan_path}")
                    predixcan_df = None
                else:
                    print(f"Loaded PrediXcan results from: {predixcan_path}")
                    print(f"  {len(predixcan_df)} regions with PrediXcan correlations")
            except Exception as e:
                print(f"Warning: Failed to load PrediXcan results: {e}")
                predixcan_df = None
        else:
            print(f"Warning: PrediXcan file not found: {predixcan_path}")

    # Create subdirectories
    per_sample_dir = output_dir / 'per_sample'
    per_region_dir = output_dir / 'per_region'

    if plot_per_sample:
        per_sample_dir.mkdir(exist_ok=True)
    if plot_per_region:
        per_region_dir.mkdir(exist_ok=True)

    # Detect format (single-track ATAC or two-track RNA)
    result_format = _detect_result_format(results_df)

    # 1. Correlation boxplot by region (CORE PLOT) - with optional PrediXcan comparison
    print("Creating correlation boxplot by region...")
    fig = plot_correlation_boxplot_by_region(
        results_df, predixcan_df=predixcan_df, model_name=model_name
    )
    for fmt in plot_formats:
        path = output_dir / f'correlation_boxplot_by_region.{fmt}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        saved_plots['summary'].append(path)
    plt.close(fig)

    # 1b. Model comparison scatter plot (if PrediXcan available)
    if predixcan_df is not None:
        print("Creating model comparison scatter plot...")
        fig_scatter = plot_model_comparison_scatter(
            results_df, predixcan_df, model_name=model_name
        )
        if fig_scatter is not None:
            for fmt in plot_formats:
                path = output_dir / f'model_comparison_scatter.{fmt}'
                fig_scatter.savefig(path, dpi=dpi, bbox_inches='tight')
                saved_plots['summary'].append(path)
            plt.close(fig_scatter)

    # 2. Correlation distribution histogram (CORE PLOT)
    print("Creating correlation distribution...")
    if result_format == 'multi_tissue':
        # For multi-tissue ATAC: Skip detailed histogram, just add a note
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 
                'Multi-Tissue ATAC Inference\n\n'
                'Correlation distributions for each tissue are available in:\n'
                'per_tissue/<tissue>_results.csv\n\n'
                'Summary statistics are in:\n'
                'tissue_summary.csv',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Multi-Tissue ATAC Results', fontsize=16, fontweight='bold')
    elif result_format == 'multi_track':
        # For multi-track CSV: Skip detailed histogram, just add a note
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get track columns
        track_cols = [col for col in results_df.columns if col.startswith('pearson_corr_ENCFF')]
        n_tracks = len(track_cols)
        
        ax.text(0.5, 0.5, 
                f'Multi-Track Inference ({n_tracks} tracks)\n\n'
                'Correlation distributions for each track are available in:\n'
                'per_track/<track>_results.csv\n\n'
                'Summary statistics are in:\n'
                'track_summary.csv',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Multi-Track CSV Results', fontsize=16, fontweight='bold')
    elif result_format == 'two_track':
        # For RNA: show both encode and gtex distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Encode distribution
        encode_corr = results_df['pearson_corr_encode'].values
        axes[0].hist(encode_corr, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0].axvline(np.nanmean(encode_corr), color='red', linestyle='--', 
                       label=f'Mean: {np.nanmean(encode_corr):.3f}')
        axes[0].set_xlabel('Pearson r')
        axes[0].set_ylabel('Number of Regions')
        axes[0].set_title('Encode Combined Track Correlation Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # GTEx distribution
        gtex_corr = results_df['pearson_corr_gtex'].values
        axes[1].hist(gtex_corr, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
        axes[1].axvline(np.nanmean(gtex_corr), color='red', linestyle='--',
                       label=f'Mean: {np.nanmean(gtex_corr):.3f}')
        axes[1].set_xlabel('Pearson r')
        axes[1].set_ylabel('Number of Regions')
        axes[1].set_title('GTEx Reference Track Correlation Distribution')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
    else:
        # For ATAC: single distribution (backward compatible)
        fig = plot_distribution(
            results_df['pearson_corr'].values,
            title='Pearson Correlation Distribution Across Regions',
            xlabel='Pearson r',
            ylabel='Number of Regions'
        )
    
    for fmt in plot_formats:
        path = output_dir / f'correlation_distribution.{fmt}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        saved_plots['summary'].append(path)
    plt.close(fig)

    # 3. Multi-panel summary plot (CORE PLOT)
    print("Creating 6-panel summary plot...")
    fig = plot_inference_summary_6panel(results_df, predictions_dict)
    for fmt in plot_formats:
        path = output_dir / f'summary_6panel.{fmt}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        saved_plots['summary'].append(path)
    plt.close(fig)

    # 4. Per-sample plots (CORE - PNG only for space)
    if plot_per_sample:
        # For multi-tissue, use first tissue as example
        sample_tissue = None
        sample_track = None
        if result_format == 'multi_tissue':
            # Get first tissue from predictions_dict
            first_region = next(iter(predictions_dict.values()))
            if 'tissue_list' in first_region:
                sample_tissue = first_region['tissue_list'][0]
                print(f"Creating per-sample correlation plots (using tissue {sample_tissue} as example)...")
            else:
                print("Warning: Multi-tissue format detected but no tissue_list found")
        elif result_format == 'multi_track':
            # Get first track from predictions_dict
            first_region = next(iter(predictions_dict.values()))
            if 'track_list' in first_region:
                sample_track = first_region['track_list'][0]
                print(f"Creating per-sample correlation plots (using track {sample_track} as example)...")
            elif 'predictions_per_track' in first_region:
                sample_track = list(first_region['predictions_per_track'].keys())[0]
                print(f"Creating per-sample correlation plots (using track {sample_track} as example)...")
            else:
                print("Warning: Multi-track format detected but no track_list found")
        else:
            print("Creating per-sample correlation plots...")
        
        sample_paths = create_per_sample_plots(
            predictions_dict,
            per_sample_dir,
            formats=['png'],  # PNG only to save space
            dpi=dpi,
            tissue=sample_tissue,
            track=sample_track
        )
        saved_plots['per_sample'].extend(sample_paths)

    # 5. Per-region plots (OPTIONAL - PNG only for space)
    if plot_per_region:
        # For multi-tissue, use first tissue as example
        region_tissue = None
        region_track = None
        if result_format == 'multi_tissue':
            # Get first tissue from predictions_dict
            first_region = next(iter(predictions_dict.values()))
            if 'tissue_list' in first_region:
                region_tissue = first_region['tissue_list'][0]
                print(f"Creating per-region correlation plots (using tissue {region_tissue} as example, this may take a while for {len(predictions_dict)} regions)...")
            else:
                print("Warning: Multi-tissue format detected but no tissue_list found")
        elif result_format == 'multi_track':
            # Get first track from predictions_dict
            first_region = next(iter(predictions_dict.values()))
            if 'track_list' in first_region:
                region_track = first_region['track_list'][0]
                print(f"Creating per-region correlation plots (using track {region_track} as example, this may take a while for {len(predictions_dict)} regions)...")
            elif 'predictions_per_track' in first_region:
                region_track = list(first_region['predictions_per_track'].keys())[0]
                print(f"Creating per-region correlation plots (using track {region_track} as example, this may take a while for {len(predictions_dict)} regions)...")
            else:
                print("Warning: Multi-track format detected but no track_list found")
        else:
            print(f"Creating per-region correlation plots (this may take a while for {len(predictions_dict)} regions)...")
        
        region_paths = create_per_region_plots(
            predictions_dict,
            results_df,
            per_region_dir,
            formats=['png'],  # PNG only to save space
            dpi=dpi,
            tissue=region_tissue,
            track=region_track
        )
        saved_plots['per_region'].extend(region_paths)

    print(f"\n✅ Created {len(saved_plots['summary'])} summary plots")
    print(f"✅ Created {len(saved_plots['per_sample'])} per-sample plots")
    print(f"✅ Created {len(saved_plots['per_region'])} per-region plots")

    return saved_plots


def plot_correlation_boxplot_by_region(
    results_df: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 8),
    predixcan_df: Optional[pd.DataFrame] = None,
    model_name: str = 'Model'
) -> plt.Figure:
    """
    Create boxplot showing distribution of per-region correlations.
    Handles both single-track (ATAC) and two-track (RNA) formats.
    Optionally includes PrediXcan comparison.

    Each region has one correlation value (across all samples for that region).

    Args:
        results_df: DataFrame with correlation columns
        figsize: Figure size
        predixcan_df: Optional DataFrame with PrediXcan results (must have test_pearson column)
        model_name: Name of the model for labeling (e.g., 'AlphaGenome', 'Borzoi')

    Returns:
        Matplotlib Figure object
    """
    # Match PrediXcan results if provided
    matched_predixcan = None
    if predixcan_df is not None and 'test_pearson' in predixcan_df.columns:
        matched_results, matched_predixcan = _match_predixcan_to_results(results_df, predixcan_df)
        if len(matched_predixcan) > 0:
            predixcan_corr = matched_predixcan['test_pearson'].dropna().values
        else:
            predixcan_corr = None
            print("Warning: No matching regions found between inference results and PrediXcan results")
    else:
        predixcan_corr = None
    result_format = _detect_result_format(results_df)
    
    if result_format == 'multi_tissue':
        # Multi-tissue ATAC: Create summary message indicating per-tissue results are separate
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract tissue-specific correlation columns
        corr_cols = [col for col in results_df.columns if col.startswith('pearson_corr_')]
        tissue_corr_cols = [col for col in corr_cols 
                            if any(col.startswith(f'pearson_corr_{prefix}') 
                                   for prefix in ['EFO', 'CL', 'UBERON'])]
        
        if len(tissue_corr_cols) == 0:
            ax.text(0.5, 0.5, 'No tissue correlation data found',
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Extract tissue names and mean correlations
        tissue_data = []
        for col in sorted(tissue_corr_cols):
            tissue_name = col.replace('pearson_corr_', '').replace('_', ':')
            correlations = results_df[col].dropna().values
            if len(correlations) > 0:
                tissue_data.append({
                    'tissue': tissue_name,
                    'mean': np.mean(correlations),
                    'median': np.median(correlations),
                    'std': np.std(correlations)
                })
        
        if len(tissue_data) == 0:
            ax.text(0.5, 0.5, 'No valid correlation data',
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Create bar chart of mean correlations per tissue
        tissues = [d['tissue'] for d in tissue_data]
        means = [d['mean'] for d in tissue_data]
        stds = [d['std'] for d in tissue_data]
        
        # Color bars by mean correlation value
        colors = []
        for mean_val in means:
            if mean_val > 0.7:
                colors.append('lightgreen')
            elif mean_val > 0.5:
                colors.append('lightyellow')
            elif mean_val > 0.3:
                colors.append('lightcoral')
            else:
                colors.append('lightgray')
        
        bars = ax.bar(range(len(tissues)), means, yerr=stds, 
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        
        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, linewidth=1, label='r=0.3')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='r=0.5')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1, label='r=0.7')
        
        # Labels
        ax.set_ylabel('Mean Pearson Correlation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tissue Ontology', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Multi-Tissue ATAC Inference Results\n'
            f'Mean Correlation per Tissue ({len(tissues)} tissues)',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Set x-axis labels
        ax.set_xticks(range(len(tissues)))
        ax.set_xticklabels(tissues, rotation=45, ha='right', fontsize=8)
        
        # Add summary stats
        overall_mean = np.mean(means)
        summary_text = (
            f"Tissues: {len(tissues)}\n"
            f"Overall Mean: {overall_mean:.3f}\n"
            f"Range: [{min(means):.3f}, {max(means):.3f}]\n\n"
            f"💡 Per-tissue details in:\n"
            f"   per_tissue/ folder"
        )
        ax.text(
            0.98, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return fig
    
    elif result_format == 'multi_track':
        # Multi-track CSV: Create summary bar chart showing per-track mean correlations
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract track-specific correlation columns
        corr_cols = [col for col in results_df.columns if col.startswith('pearson_corr_ENCFF')]
        
        if len(corr_cols) == 0:
            ax.text(0.5, 0.5, 'No track correlation data found',
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Extract track names and mean correlations
        track_data = []
        for col in sorted(corr_cols):
            track_name = col.replace('pearson_corr_', '')
            correlations = results_df[col].dropna().values
            if len(correlations) > 0:
                track_data.append({
                    'track': track_name,
                    'mean': np.mean(correlations),
                    'median': np.median(correlations),
                    'std': np.std(correlations)
                })
        
        if len(track_data) == 0:
            ax.text(0.5, 0.5, 'No valid correlation data',
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by mean correlation (descending)
        track_data.sort(key=lambda x: x['mean'], reverse=True)
        
        # Create bar chart of mean correlations per track
        tracks = [d['track'] for d in track_data]
        means = [d['mean'] for d in track_data]
        stds = [d['std'] for d in track_data]
        
        # Color bars by mean correlation value
        colors = []
        for mean_val in means:
            if mean_val > 0.7:
                colors.append('lightgreen')
            elif mean_val > 0.5:
                colors.append('lightyellow')
            elif mean_val > 0.3:
                colors.append('lightcoral')
            elif mean_val > 0:
                colors.append('lightgray')
            else:
                colors.append('lightblue')  # Negative correlations
        
        bars = ax.bar(range(len(tracks)), means, yerr=stds, 
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        
        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, linewidth=1, label='r=0.3')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='r=0.5')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1, label='r=0.7')
        
        # Labels
        ax.set_ylabel('Mean Pearson Correlation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Track ID', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Multi-Track Inference Results\n'
            f'Mean Correlation per Track ({len(tracks)} tracks)',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Set x-axis labels
        ax.set_xticks(range(len(tracks)))
        ax.set_xticklabels(tracks, rotation=45, ha='right', fontsize=8)
        
        # Add summary stats
        overall_mean = np.mean(means)
        summary_text = (
            f"n = {len(results_df)} regions\n"
            f"Tracks: {len(tracks)}\n"
            f"Overall Mean: {overall_mean:.3f}\n"
            f"Best: {tracks[0]} ({means[0]:.3f})\n\n"
            f"Details in:\n"
            f"   per_track/ folder"
        )
        ax.text(
            0.98, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return fig
    
    elif result_format == 'two_track':
        # Two-track RNA: side-by-side boxplots with optional PrediXcan
        fig, ax = plt.subplots(figsize=figsize)

        corr_encode = results_df['pearson_corr_encode'].dropna().values
        corr_gtex = results_df['pearson_corr_gtex'].dropna().values

        if len(corr_encode) == 0 and len(corr_gtex) == 0:
            ax.text(0.5, 0.5, 'No valid correlation data',
                    ha='center', va='center', fontsize=14)
            return fig

        # Create side-by-side boxplots (include PrediXcan if available)
        if predixcan_corr is not None and len(predixcan_corr) > 0:
            bp = ax.boxplot(
                [corr_encode, corr_gtex, predixcan_corr],
                labels=[f'{model_name} (Encode)', f'{model_name} (GTEx)', 'PrediXcan'],
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=10),
                medianprops=dict(color='darkblue', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )

            # Color boxes
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('lightcoral')
            bp['boxes'][1].set_alpha(0.7)
            bp['boxes'][2].set_facecolor('lightsalmon')
            bp['boxes'][2].set_alpha(0.7)

            title_text = (
                f'Per-Region Correlations: {model_name} vs PrediXcan\n'
                f'(Each region: correlation across samples)'
            )

            # Add summary stats as text
            summary_text = f"n = {len(corr_encode)} regions\n\n"
            if len(corr_encode) > 0:
                summary_text += (
                    f"{model_name} (Encode):\n"
                    f"  Mean: {np.mean(corr_encode):.3f} ± {np.std(corr_encode):.3f}\n\n"
                )
            if len(corr_gtex) > 0:
                summary_text += (
                    f"{model_name} (GTEx):\n"
                    f"  Mean: {np.mean(corr_gtex):.3f} ± {np.std(corr_gtex):.3f}\n\n"
                )
            summary_text += (
                f"PrediXcan:\n"
                f"  Mean: {np.mean(predixcan_corr):.3f} ± {np.std(predixcan_corr):.3f}"
            )
        else:
            bp = ax.boxplot(
                [corr_encode, corr_gtex],
                labels=['Encode Combined', 'GTEx Reference'],
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=10),
                medianprops=dict(color='darkblue', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )

            # Color boxes
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('lightcoral')
            bp['boxes'][1].set_alpha(0.7)

            title_text = (
                f'Distribution of Per-Region Correlations (Two Tracks)\n'
                f'(Each region: correlation across samples)'
            )

            # Add summary stats as text
            summary_text = f"n = {len(corr_encode)} regions\n\n"
            if len(corr_encode) > 0:
                summary_text += (
                    f"Encode Combined:\n"
                    f"  Mean: {np.mean(corr_encode):.3f} ± {np.std(corr_encode):.3f}\n"
                    f"  Median: {np.median(corr_encode):.3f}\n\n"
                )
            if len(corr_gtex) > 0:
                summary_text += (
                    f"GTEx Reference:\n"
                    f"  Mean: {np.mean(corr_gtex):.3f} ± {np.std(corr_gtex):.3f}\n"
                    f"  Median: {np.median(corr_gtex):.3f}"
                )

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1)

        ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        ax.text(
            0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
    else:
        # Single-track ATAC: original behavior with optional PrediXcan comparison
        fig, ax = plt.subplots(figsize=figsize)

        correlations = results_df['pearson_corr'].dropna().values

        if len(correlations) == 0:
            ax.text(0.5, 0.5, 'No valid correlation data',
                    ha='center', va='center', fontsize=14)
            return fig

        # Prepare data for boxplot
        if predixcan_corr is not None and len(predixcan_corr) > 0:
            # Side-by-side comparison with PrediXcan
            bp = ax.boxplot(
                [correlations, predixcan_corr],
                labels=[model_name, 'PrediXcan'],
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=10),
                medianprops=dict(color='darkblue', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )

            # Color boxes: model in blue, PrediXcan in orange
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('lightsalmon')
            bp['boxes'][1].set_alpha(0.7)

            title_text = (
                f'Distribution of Per-Region Correlations: {model_name} vs PrediXcan\n'
                f'(Each region: correlation across samples)'
            )

            # Summary stats for both
            summary_text = (
                f"n = {len(correlations)} regions\n\n"
                f"{model_name}:\n"
                f"  Mean: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}\n"
                f"  Median: {np.median(correlations):.3f}\n\n"
                f"PrediXcan:\n"
                f"  Mean: {np.mean(predixcan_corr):.3f} ± {np.std(predixcan_corr):.3f}\n"
                f"  Median: {np.median(predixcan_corr):.3f}"
            )
        else:
            # Single boxplot (original behavior)
            bp = ax.boxplot(
                [correlations],
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=10),
                medianprops=dict(color='darkblue', linewidth=2),
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )

            # Color box by mean correlation
            mean_corr = np.mean(correlations)
            if mean_corr > 0.7:
                bp['boxes'][0].set_facecolor('lightgreen')
            elif mean_corr > 0.5:
                bp['boxes'][0].set_facecolor('lightyellow')
            elif mean_corr > 0.3:
                bp['boxes'][0].set_facecolor('lightcoral')
            else:
                bp['boxes'][0].set_facecolor('lightgray')

            ax.set_xticklabels(['All Regions'])

            title_text = (
                f'Distribution of Per-Region Correlations\n'
                f'(Each region: correlation across samples)'
            )

            # Summary stats
            summary_text = (
                f"n = {len(correlations)} regions\n"
                f"Mean: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}\n"
                f"Median: {np.median(correlations):.3f}\n"
                f"Range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]"
            )

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, linewidth=1, label='r=0.3')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='r=0.5')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1, label='r=0.7')

        # Labels
        ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

        ax.text(
            0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_model_comparison_scatter(
    results_df: pd.DataFrame,
    predixcan_df: pd.DataFrame,
    model_name: str = 'Model',
    figsize: Tuple[float, float] = (10, 10)
) -> Optional[plt.Figure]:
    """
    Create scatter plot comparing per-region correlations between model and PrediXcan.

    X-axis: PrediXcan test_pearson
    Y-axis: Model pearson_corr
    Includes diagonal line, correlation coefficient, and color by region.

    Args:
        results_df: Inference results DataFrame with pearson_corr column
        predixcan_df: PrediXcan results DataFrame with test_pearson column
        model_name: Name of the model for labeling (e.g., 'AlphaGenome', 'Borzoi')
        figsize: Figure size

    Returns:
        Matplotlib Figure object, or None if no matching regions found
    """
    if 'test_pearson' not in predixcan_df.columns:
        print("Warning: PrediXcan DataFrame missing 'test_pearson' column")
        return None

    # Match regions between results and PrediXcan
    matched_results, matched_predixcan = _match_predixcan_to_results(results_df, predixcan_df)

    if len(matched_results) == 0:
        print("Warning: No matching regions found between inference and PrediXcan results")
        return None

    # Detect result format to get correct correlation column
    result_format = _detect_result_format(matched_results)

    if result_format == 'two_track':
        # For two-track RNA, use encode track by default
        model_corr = matched_results['pearson_corr_encode'].values
        ylabel = f'{model_name} Pearson r (Encode track)'
    elif result_format == 'multi_track':
        # For multi-track, use first track
        track_cols = [c for c in matched_results.columns if c.startswith('pearson_corr_ENCFF')]
        if track_cols:
            model_corr = matched_results[track_cols[0]].values
            track_name = track_cols[0].replace('pearson_corr_', '')
            ylabel = f'{model_name} Pearson r ({track_name})'
        else:
            print("Warning: No track correlation columns found")
            return None
    elif result_format == 'multi_tissue':
        # For multi-tissue, use first tissue
        tissue_cols = [c for c in matched_results.columns if c.startswith('pearson_corr_') and
                      any(c.startswith(f'pearson_corr_{p}') for p in ['EFO', 'CL', 'UBERON'])]
        if tissue_cols:
            model_corr = matched_results[tissue_cols[0]].values
            tissue_name = tissue_cols[0].replace('pearson_corr_', '').replace('_', ':')
            ylabel = f'{model_name} Pearson r ({tissue_name})'
        else:
            print("Warning: No tissue correlation columns found")
            return None
    else:
        # Single-track ATAC
        model_corr = matched_results['pearson_corr'].values
        ylabel = f'{model_name} Pearson r'

    predixcan_corr = matched_predixcan['test_pearson'].values

    # Remove NaN pairs
    valid_mask = ~(np.isnan(model_corr) | np.isnan(predixcan_corr))
    model_corr = model_corr[valid_mask]
    predixcan_corr = predixcan_corr[valid_mask]

    if len(model_corr) == 0:
        print("Warning: No valid correlation pairs after removing NaN values")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(
        predixcan_corr, model_corr,
        c='steelblue', alpha=0.6, edgecolors='darkblue', linewidths=0.5, s=50
    )

    # Diagonal line (y = x)
    min_val = min(predixcan_corr.min(), model_corr.min()) - 0.1
    max_val = max(predixcan_corr.max(), model_corr.max()) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x', linewidth=1.5)

    # Calculate correlation between model and PrediXcan correlations
    if len(model_corr) > 2:
        r_value, p_value = pearsonr(predixcan_corr, model_corr)
    else:
        r_value, p_value = np.nan, np.nan

    # Reference lines at 0
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3, linewidth=1)

    # Labels
    ax.set_xlabel('PrediXcan test_pearson', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(
        f'Model Comparison: {model_name} vs PrediXcan\n'
        f'Per-Region Correlation Comparison',
        fontsize=14, fontweight='bold', pad=15
    )

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Summary statistics
    model_better = np.sum(model_corr > predixcan_corr)
    predixcan_better = np.sum(predixcan_corr > model_corr)
    mean_diff = np.mean(model_corr - predixcan_corr)

    summary_text = (
        f"n = {len(model_corr)} regions\n\n"
        f"Correlation (r): {r_value:.3f}\n"
        f"p-value: {p_value:.2e}\n\n"
        f"{model_name} > PrediXcan: {model_better}\n"
        f"PrediXcan > {model_name}: {predixcan_better}\n\n"
        f"Mean diff: {mean_diff:+.3f}"
    )

    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_inference_summary_6panel(
    results_df: pd.DataFrame,
    predictions_dict: Dict[str, Dict],
    figsize: Tuple[float, float] = (20, 12)
) -> plt.Figure:
    """
    Create 6-panel summary plot for inference results.
    Handles both single-track (ATAC) and two-track (RNA) formats.

    Args:
        results_df: DataFrame with inference results
        predictions_dict: Dict with predictions and observations
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    result_format = _detect_result_format(results_df)
    
    # For multi-tissue, show informational message instead
    if result_format == 'multi_tissue':
        fig, ax = plt.subplots(figsize=(12, 8))
        info_text = (
            'Multi-Tissue ATAC Inference Results\n\n'
            'Summary visualizations are not generated for multi-tissue mode\n'
            'due to the large number of tissues.\n\n'
            'Detailed results are available in:\n'
            '  • tissue_summary.csv - Overall statistics per tissue\n'
            '  • per_tissue/<tissue>_results.csv - Per-region results for each tissue\n'
            '  • predictions/<region_id>.npz - Per-region predictions\n\n'
            'Use the correlation_boxplot_by_region plot to see\n'
            'mean correlations across all tissues.'
        )
        ax.text(0.5, 0.5, info_text,
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Multi-Tissue ATAC Results', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    # For multi-track CSV, show informational message instead
    if result_format == 'multi_track':
        track_cols = [col for col in results_df.columns if col.startswith('pearson_corr_ENCFF')]
        n_tracks = len(track_cols)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        info_text = (
            f'Multi-Track Inference Results ({n_tracks} tracks)\n\n'
            'Summary visualizations are not generated for multi-track mode\n'
            'due to the large number of tracks.\n\n'
            'Detailed results are available in:\n'
            '  • track_summary.csv - Overall statistics per track\n'
            '  • per_track/<track>_results.csv - Per-region results for each track\n'
            '  • predictions/<region_id>.npz - Per-region predictions\n\n'
            'Use the correlation_boxplot_by_region plot to see\n'
            'mean correlations across all tracks.'
        )
        ax.text(0.5, 0.5, info_text,
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Multi-Track CSV Results', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    title_suffix = ' (RNA Two-Track)' if result_format == 'two_track' else ''
    fig.suptitle(f'Inference Results Summary{title_suffix}', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Panel 1: Correlation distribution
    ax = axes[0]
    if result_format == 'two_track':
        # Show encode track (primary)
        correlations = results_df['pearson_corr_encode'].dropna()
        color = 'skyblue'
        title = 'Correlation Distribution (Encode)'
    else:
        correlations = results_df['pearson_corr'].dropna()
        color = 'skyblue'
        title = 'Correlation Distribution'
        
    if len(correlations) > 0:
        ax.hist(correlations, bins=min(20, len(correlations)//2 + 1),
                alpha=0.7, edgecolor='black', color=color)
        ax.axvline(correlations.mean(), color='red', linestyle='--',
                   label=f'Mean: {correlations.mean():.3f}')
        ax.axvline(correlations.median(), color='orange', linestyle='--',
                   label=f'Median: {correlations.median():.3f}')
        ax.set_xlabel('Pearson r')
        ax.set_ylabel('Number of Regions')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    # Panel 2: Predicted vs Observed (all data)
    ax = axes[1]
    all_pred = []
    all_obs = []
    for region_data in predictions_dict.values():
        # Use helper function to get predictions
        preds = _get_predictions_array(region_data, track='encode')
        all_pred.extend(preds)
        all_obs.extend(region_data['observed'])

    if len(all_pred) > 0:
        ax.scatter(all_obs, all_pred, alpha=0.4, s=20, edgecolor='k', linewidth=0.3)

        # Identity line
        lim_min = min(min(all_obs), min(all_pred))
        lim_max = max(max(all_obs), max(all_pred))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.7, label='y=x')

        # Overall correlation
        if len(all_pred) > 1:
            r, p = pearsonr(all_obs, all_pred)
            ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(all_pred)}',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        track_label = ' (Encode)' if result_format == 'two_track' else ''
        ax.set_xlabel('Observed')
        ax.set_ylabel(f'Predicted{track_label}')
        ax.set_title(f'All Predictions vs Observed{track_label}')
        ax.legend()
        ax.grid(alpha=0.3)

    # Panel 3: Sample size distribution
    ax = axes[2]
    sample_counts = results_df['n_samples'].dropna()
    if len(sample_counts) > 0:
        ax.hist(sample_counts, bins=min(15, len(sample_counts.unique())),
                alpha=0.7, edgecolor='black', color='lightgreen')
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Number of Regions')
        ax.set_title(f'Sample Size Distribution\n(Mean: {sample_counts.mean():.1f})')
        ax.grid(alpha=0.3)

    # Panel 4: Mean prediction vs observation by region
    ax = axes[3]
    # For two-track, use encode predictions
    mean_pred_col = 'mean_pred_encode' if result_format == 'two_track' else 'mean_pred'
    
    if mean_pred_col in results_df.columns and 'mean_obs' in results_df.columns:
        valid_mask = results_df[mean_pred_col].notna() & results_df['mean_obs'].notna()
        if valid_mask.any():
            ax.scatter(
                results_df.loc[valid_mask, 'mean_obs'],
                results_df.loc[valid_mask, mean_pred_col],
                alpha=0.6, s=50, edgecolor='k', linewidth=0.5, color='coral'
            )

            # Identity line
            obs_vals = results_df.loc[valid_mask, 'mean_obs'].values
            pred_vals = results_df.loc[valid_mask, mean_pred_col].values
            lim_min = min(obs_vals.min(), pred_vals.min())
            lim_max = max(obs_vals.max(), pred_vals.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.7)

            track_label = ' (Encode)' if result_format == 'two_track' else ''
            ax.set_xlabel('Mean Observed (per region)')
            ax.set_ylabel(f'Mean Predicted{track_label} (per region)')
            ax.set_title(f'Region-Level Means{track_label}')
            ax.grid(alpha=0.3)

    # Panel 5: Correlation vs sample size
    ax = axes[4]
    corr_col = 'pearson_corr_encode' if result_format == 'two_track' else 'pearson_corr'
    valid_mask = results_df[corr_col].notna() & results_df['n_samples'].notna()
    
    if valid_mask.any():
        ax.scatter(
            results_df.loc[valid_mask, 'n_samples'],
            results_df.loc[valid_mask, corr_col],
            alpha=0.6, s=50, edgecolor='k', linewidth=0.5, color='mediumpurple'
        )
        ax.set_xlabel('Number of Samples')
        track_label = ' (Encode)' if result_format == 'two_track' else ''
        ax.set_ylabel(f'Pearson Correlation{track_label}')
        ax.set_title(f'Correlation vs Sample Size{track_label}')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(alpha=0.3)

    # Panel 6: Summary statistics table
    ax = axes[5]
    ax.axis('off')

    # Calculate summary statistics
    stats_text = "Summary Statistics\n" + "="*40 + "\n\n"
    stats_text += f"Total Regions: {len(results_df)}\n"
    stats_text += f"Total Predictions: {sum(results_df['n_samples'])}\n\n"

    if result_format == 'two_track':
        # Two-track summary
        corr_encode = results_df['pearson_corr_encode'].dropna()
        corr_gtex = results_df['pearson_corr_gtex'].dropna()
        
        if len(corr_encode) > 0:
            stats_text += "Encode Combined Track:\n"
            stats_text += f"  Mean: {corr_encode.mean():.3f} ± {corr_encode.std():.3f}\n"
            stats_text += f"  Median: {corr_encode.median():.3f}\n"
            stats_text += f"  Range: [{corr_encode.min():.3f}, {corr_encode.max():.3f}]\n"
            stats_text += f"  >0.5: {(corr_encode > 0.5).sum()} ({(corr_encode > 0.5).sum()/len(corr_encode)*100:.1f}%)\n\n"
        
        if len(corr_gtex) > 0:
            stats_text += "GTEx Reference Track:\n"
            stats_text += f"  Mean: {corr_gtex.mean():.3f} ± {corr_gtex.std():.3f}\n"
            stats_text += f"  Median: {corr_gtex.median():.3f}\n"
            stats_text += f"  Range: [{corr_gtex.min():.3f}, {corr_gtex.max():.3f}]\n"
            stats_text += f"  >0.5: {(corr_gtex > 0.5).sum()} ({(corr_gtex > 0.5).sum()/len(corr_gtex)*100:.1f}%)\n"
    else:
        # Single-track summary
        if len(correlations) > 0:
            stats_text += "Correlations:\n"
            stats_text += f"  Mean: {correlations.mean():.3f} ± {correlations.std():.3f}\n"
            stats_text += f"  Median: {correlations.median():.3f}\n"
            stats_text += f"  Range: [{correlations.min():.3f}, {correlations.max():.3f}]\n"
            stats_text += f"  >0.5: {(correlations > 0.5).sum()} ({(correlations > 0.5).sum()/len(correlations)*100:.1f}%)\n"
            stats_text += f"  >0.7: {(correlations > 0.7).sum()} ({(correlations > 0.7).sum()/len(correlations)*100:.1f}%)\n\n"

        if 'mean_pred' in results_df.columns:
            stats_text += f"Mean Prediction: {results_df['mean_pred'].mean():.3f}\n"
            stats_text += f"Mean Observation: {results_df['mean_obs'].mean():.3f}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    return fig


def create_per_sample_plots(
    predictions_dict: Dict[str, Dict],
    output_dir: Path,
    formats: List[str] = ['png'],
    dpi: int = 150,
    tissue: str = None,
    track: str = None
) -> List[Path]:
    """
    Create one correlation plot per sample.

    Each plot shows predicted vs observed across all regions for that sample.

    Args:
        predictions_dict: Dict mapping region_id -> {predictions, observed, sample_ids}
        output_dir: Directory to save plots
        formats: List of formats ['png', 'pdf']
        dpi: DPI for saved figures
        tissue: For multi-tissue data, which tissue to use. If None, uses first tissue.
        track: For multi-track data, which track to use. If None, uses first track.

    Returns:
        List of saved file paths
    """
    # Reorganize data by sample
    sample_data = {}

    for region_id, region_data in predictions_dict.items():
        # Use helper function to get predictions (handles all formats)
        preds = _get_predictions_array(region_data, track=track or 'encode', tissue=tissue)
        obs = region_data['observed']
        samples = region_data['sample_ids']

        for sample_id, pred, ob in zip(samples, preds, obs):
            if sample_id not in sample_data:
                sample_data[sample_id] = {'predicted': [], 'observed': []}
            sample_data[sample_id]['predicted'].append(pred)
            sample_data[sample_id]['observed'].append(ob)

    saved_paths = []

    # Create plot for each sample
    for sample_id, data in sample_data.items():
        predicted = np.array(data['predicted'])
        observed = np.array(data['observed'])

        # Create plot
        fig = plot_prediction_vs_observed(
            y_observed=observed,
            y_predicted=predicted,
            title=f'Sample: {sample_id}',
            xlabel='Observed (across regions)',
            ylabel='Predicted (across regions)',
            show_correlation=True,
            show_identity=True,
            figsize=(8, 8)
        )

        # Save in requested formats
        safe_sample_id = str(sample_id).replace('/', '_').replace(':', '_')
        for fmt in formats:
            path = output_dir / f'sample_{safe_sample_id}_correlation.{fmt}'
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            saved_paths.append(path)

        plt.close(fig)

    return saved_paths


def create_per_region_plots(
    predictions_dict: Dict[str, Dict],
    results_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str] = ['png'],
    dpi: int = 150,
    tissue: str = None,
    track: str = None
) -> List[Path]:
    """
    Create one correlation plot per region.

    Each plot shows predicted vs observed across all samples for that region.

    Args:
        predictions_dict: Dict mapping region_id -> {predictions, observed, sample_ids}
        results_df: DataFrame with region metadata
        output_dir: Directory to save plots
        formats: List of formats ['png', 'pdf']
        dpi: DPI for saved figures
        tissue: For multi-tissue data, which tissue to use. If None, uses first tissue.
        track: For multi-track data, which track to use. If None, uses first track.

    Returns:
        List of saved file paths
    """
    saved_paths = []

    for region_id, region_data in predictions_dict.items():
        # Use helper function to get predictions (handles all formats)
        predicted = _get_predictions_array(region_data, track=track or 'encode', tissue=tissue)
        observed = region_data['observed']

        # Get region name from results_df
        region_info = results_df[results_df['region_id'] == region_id]
        if len(region_info) > 0:
            region_name = region_info.iloc[0].get('region_name', region_id)
            chr_info = region_info.iloc[0].get('chr', '')
        else:
            region_name = str(region_id)
            chr_info = ''

        # Create title
        title = f'Region: {region_name}'
        if chr_info:
            title += f' (Chr {chr_info})'

        # Create plot
        fig = plot_prediction_vs_observed(
            y_observed=observed,
            y_predicted=predicted,
            title=title,
            xlabel='Observed (across samples)',
            ylabel='Predicted (across samples)',
            show_correlation=True,
            show_identity=True,
            figsize=(8, 8)
        )

        # Save in requested formats
        safe_region_name = str(region_name).replace('/', '_').replace(':', '_')
        for fmt in formats:
            path = output_dir / f'region_{safe_region_name}_correlation.{fmt}'
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            saved_paths.append(path)

        plt.close(fig)

    return saved_paths