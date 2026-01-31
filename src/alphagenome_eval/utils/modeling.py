"""
Statistical modeling and machine learning utilities for AlphaGenome evaluation.

This module provides unified functions for PrediXcan analysis, including:
- ElasticNet model training with cross-validation or validation sets
- Model evaluation metrics (R², Pearson correlation)
- Complete PrediXcan pipeline for genomic regions
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import PredefinedSplit, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr, spearmanr

# Suppress convergence warnings by default
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def train_elasticnet(
    X_train: np.ndarray,
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    l1_ratio: float = 0.5,
    alphas: Optional[np.ndarray] = None,
    max_iter: int = 2000,
    random_state: int = 42,
    cv_folds: int = 5,
    min_samples: int = 10
) -> Dict[str, Any]:
    """
    Train ElasticNet model with optional validation set for hyperparameter tuning.

    This function unifies both LCL and ROSMAP PrediXcan approaches:
    - If X_val is provided: Use explicit validation set for hyperparameter search
    - If X_val is None: Use internal cross-validation on training set

    Args:
        X_train: Training genotype matrix (n_samples, n_variants)
        y_train: Training expression/accessibility values
        X_val: Optional validation genotype matrix
        y_val: Optional validation expression/accessibility values
        X_test: Optional test genotype matrix
        y_test: Optional test expression/accessibility values
        l1_ratio: ElasticNet mixing parameter (0.5 = balanced Ridge/Lasso)
        alphas: Array of alpha values to test (None = auto-generate)
        max_iter: Maximum iterations for convergence
        random_state: Random seed for reproducibility
        cv_folds: Number of CV folds if using internal CV
        min_samples: Minimum samples required for training

    Returns:
        Dictionary with:
            - model: Trained ElasticNet model (or None if failed)
            - train_r2, val_r2, test_r2: R² scores
            - train_pearson, val_pearson, test_pearson: Pearson correlations
            - train_pred, val_pred, test_pred: Predictions
            - y_train, y_val, y_test: Actual values
            - n_variants: Number of variants used
            - n_active_variants: Number of non-zero coefficients
            - optimal_alpha: Selected regularization parameter
            - status: 'success', 'no_variants', 'insufficient_samples', or error message

    Example:
        >>> # LCL approach: internal CV
        >>> result = train_elasticnet(X_train, y_train, X_test=X_test, y_test=y_test)

        >>> # ROSMAP approach: explicit validation set
        >>> result = train_elasticnet(
        ...     X_train, y_train, X_val, y_val, X_test, y_test
        ... )
        >>> print(f"Test Pearson: {result['test_pearson']:.3f}")
    """
    # Convert pandas Series to numpy arrays
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if y_val is not None and hasattr(y_val, 'values'):
        y_val = y_val.values
    if y_test is not None and hasattr(y_test, 'values'):
        y_test = y_test.values

    # Check for no variants
    if X_train.shape[1] == 0:
        return _create_fallback_result(
            y_train, y_val, y_test,
            status='no_variants'
        )

    # Check for insufficient samples
    if X_train.shape[0] < min_samples:
        return _create_fallback_result(
            y_train, y_val, y_test,
            status='insufficient_samples'
        )

    # Check validation set if provided
    if X_val is not None and (X_val.shape[0] < 5 or y_val is None):
        return _create_fallback_result(
            y_train, y_val, y_test,
            status='insufficient_validation_samples'
        )

    # Generate alpha range if not provided
    if alphas is None:
        alphas = np.logspace(-3, 1, 20)  # 0.001 to 10

    try:
        # ===== Case 1: Explicit validation set (ROSMAP approach) =====
        if X_val is not None:
            best_alpha = None
            best_val_score = -np.inf
            best_model = None
            best_val_pred = None  # Store predictions from best model

            # Grid search over alphas
            for alpha in alphas:
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=random_state,
                    max_iter=max_iter
                )
                model.fit(X_train, y_train)

                # Evaluate on validation set
                val_pred_current = model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred_current)

                # Keep best model AND its predictions
                if val_r2 > best_val_score:
                    best_val_score = val_r2
                    best_alpha = alpha
                    best_model = model
                    best_val_pred = val_pred_current  # Save best predictions

            if best_model is None:
                return _create_fallback_result(
                    y_train, y_val, y_test,
                    status='no_convergence'
                )

            final_model = best_model
            optimal_alpha = best_alpha
            val_pred = best_val_pred  # Use predictions from best model

        # ===== Case 2: Internal CV (LCL approach) =====
        else:
            # Create validation fold for cross-validation
            test_fold = np.full(X_train.shape[0], -1)
            n_val = max(1, X_train.shape[0] // cv_folds)
            val_indices = np.random.RandomState(random_state).choice(
                X_train.shape[0], size=n_val, replace=False
            )
            test_fold[val_indices] = 0

            # Train with ElasticNetCV
            cv = PredefinedSplit(test_fold)
            cv_model = ElasticNetCV(
                cv=cv,
                l1_ratio=l1_ratio,
                alphas=alphas,
                random_state=random_state,
                max_iter=max_iter
            )
            cv_model.fit(X_train, y_train)

            # Train final model with selected alpha
            optimal_alpha = cv_model.alpha_
            final_model = ElasticNet(
                alpha=optimal_alpha,
                l1_ratio=l1_ratio,
                random_state=random_state,
                max_iter=max_iter
            )
            final_model.fit(X_train, y_train)
            
            # Capture CV predictions for validation metrics
            # Use KFold CV to get out-of-fold predictions for metrics
            kfold_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_predictions = cross_val_predict(
                ElasticNet(alpha=optimal_alpha, l1_ratio=l1_ratio, 
                          random_state=random_state, max_iter=max_iter),
                X_train, y_train, cv=kfold_cv
            )
            
            # Store CV predictions as "validation" predictions
            # This allows us to populate val_r2 and val_pearson with CV metrics
            X_val = X_train
            y_val = y_train
            val_pred = cv_predictions

        # ===== Make predictions =====
        train_pred = final_model.predict(X_train)
        # val_pred already set in CV case, otherwise predict on explicit validation set
        if X_val is not None and 'val_pred' not in locals():
            val_pred = final_model.predict(X_val)
        test_pred = final_model.predict(X_test) if X_test is not None else None

        # ===== Calculate metrics =====
        metrics = evaluate_predictions(
            y_train=y_train,
            train_pred=train_pred,
            y_val=y_val,
            val_pred=val_pred if 'val_pred' in locals() else None,
            y_test=y_test,
            test_pred=test_pred
        )

        # ===== Count active variants =====
        n_active_variants = np.sum(np.abs(final_model.coef_) > 1e-6)

        # ===== Return results =====
        return {
            'model': final_model,
            'status': 'success',
            'optimal_alpha': optimal_alpha,
            'n_variants': X_train.shape[1],
            'n_active_variants': n_active_variants,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            **metrics
        }

    except Exception as e:
        return _create_fallback_result(
            y_train, y_val, y_test,
            status=f'error: {str(e)}'
        )


def evaluate_predictions(
    y_train: np.ndarray,
    train_pred: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    val_pred: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    test_pred: Optional[np.ndarray] = None,
    include_spearman: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for predictions.

    Args:
        y_train: Training true values
        train_pred: Training predictions
        y_val: Validation true values (optional)
        val_pred: Validation predictions (optional)
        y_test: Test true values (optional)
        test_pred: Test predictions (optional)
        include_spearman: Also compute Spearman correlation

    Returns:
        Dictionary with R², Pearson (and optionally Spearman) for each split

    Example:
        >>> metrics = evaluate_predictions(y_train, train_pred, y_test, test_pred)
        >>> print(f"Test R²: {metrics['test_r2']:.3f}")
        >>> print(f"Test Pearson: {metrics['test_pearson']:.3f}")
    """
    metrics = {}

    # Training metrics
    metrics['train_r2'] = r2_score(y_train, train_pred)
    metrics['train_pearson'], _ = pearsonr(y_train, train_pred)
    if include_spearman:
        metrics['train_spearman'], _ = spearmanr(y_train, train_pred)

    # Handle NaN correlations
    if np.isnan(metrics['train_pearson']):
        metrics['train_pearson'] = 0.0

    # Validation metrics
    if y_val is not None and val_pred is not None:
        metrics['val_r2'] = r2_score(y_val, val_pred)
        if len(y_val) > 1:
            metrics['val_pearson'], _ = pearsonr(y_val, val_pred)
            if include_spearman:
                metrics['val_spearman'], _ = spearmanr(y_val, val_pred)
        else:
            metrics['val_pearson'] = np.nan
            if include_spearman:
                metrics['val_spearman'] = np.nan

        # Handle NaN
        if np.isnan(metrics['val_pearson']):
            metrics['val_pearson'] = 0.0
    else:
        metrics['val_r2'] = np.nan
        metrics['val_pearson'] = np.nan
        if include_spearman:
            metrics['val_spearman'] = np.nan

    # Test metrics
    if y_test is not None and test_pred is not None:
        metrics['test_r2'] = r2_score(y_test, test_pred)
        if len(y_test) > 1:
            metrics['test_pearson'], _ = pearsonr(y_test, test_pred)
            if include_spearman:
                metrics['test_spearman'], _ = spearmanr(y_test, test_pred)
        else:
            metrics['test_pearson'] = np.nan
            if include_spearman:
                metrics['test_spearman'] = np.nan

        # Handle NaN
        if np.isnan(metrics['test_pearson']):
            metrics['test_pearson'] = 0.0
    else:
        metrics['test_r2'] = np.nan
        metrics['test_pearson'] = np.nan
        if include_spearman:
            metrics['test_spearman'] = np.nan

    return metrics


def run_predixcan_for_region(
    genotype_matrix: np.ndarray,
    expression_values: Union[pd.Series, np.ndarray],
    train_samples: List[str],
    test_samples: List[str],
    sample_names: List[str],
    val_samples: Optional[List[str]] = None,
    l1_ratio: float = 0.5,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete PrediXcan pipeline for a single genomic region (peak or gene).

    This function handles sample indexing, trains the model, and returns results
    in a standardized format compatible with both LCL and ROSMAP workflows.

    Args:
        genotype_matrix: Genotype matrix (n_samples, n_variants)
        expression_values: Expression/accessibility values for all samples
        train_samples: List of training sample names
        test_samples: List of test sample names
        sample_names: List of all sample names (for indexing)
        val_samples: Optional list of validation sample names
        l1_ratio: ElasticNet mixing parameter
        random_state: Random seed
        **kwargs: Additional arguments passed to train_elasticnet

    Returns:
        Dictionary with model results and performance metrics

    Example:
        >>> result = run_predixcan_for_region(
        ...     genotypes, expression, train_samps, test_samps, all_samps
        ... )
        >>> if result['status'] == 'success':
        ...     print(f"Test Pearson: {result['test_pearson']:.3f}")
        ...     print(f"Active variants: {result['n_active_variants']}")
    """
    # Create sample index mapping
    sample_to_idx = {sample: i for i, sample in enumerate(sample_names)}

    # Get sample indices
    train_indices = [sample_to_idx[s] for s in train_samples if s in sample_to_idx]
    test_indices = [sample_to_idx[s] for s in test_samples if s in sample_to_idx]

    # Check if we have enough samples
    if len(train_indices) == 0 or len(test_indices) == 0:
        y_train = expression_values.iloc[train_indices] if hasattr(expression_values, 'iloc') else expression_values[train_indices] if len(train_indices) > 0 else np.array([])
        y_test = expression_values.iloc[test_indices] if hasattr(expression_values, 'iloc') else expression_values[test_indices] if len(test_indices) > 0 else np.array([])
        return _create_fallback_result(
            y_train, None, y_test,
            status='insufficient_samples'
        )

    # Prepare training data
    X_train = genotype_matrix[train_indices]
    X_test = genotype_matrix[test_indices]
    y_train = expression_values.iloc[train_indices] if hasattr(expression_values, 'iloc') else expression_values[train_indices]
    y_test = expression_values.iloc[test_indices] if hasattr(expression_values, 'iloc') else expression_values[test_indices]

    # Prepare validation data if provided
    X_val = None
    y_val = None
    if val_samples is not None:
        val_indices = [sample_to_idx[s] for s in val_samples if s in sample_to_idx]
        if len(val_indices) > 0:
            X_val = genotype_matrix[val_indices]
            y_val = expression_values.iloc[val_indices] if hasattr(expression_values, 'iloc') else expression_values[val_indices]

    # Train model
    result = train_elasticnet(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        l1_ratio=l1_ratio,
        random_state=random_state,
        **kwargs
    )

    return result


def calculate_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate pairwise correlation matrix.

    Args:
        data: DataFrame with samples as rows and features as columns
        method: 'pearson' or 'spearman'

    Returns:
        Correlation matrix as DataFrame

    Example:
        >>> corr_matrix = calculate_correlation_matrix(predictions_df)
        >>> print(corr_matrix.loc['observed', 'predicted'])
    """
    if method == 'pearson':
        return data.corr(method='pearson')
    elif method == 'spearman':
        return data.corr(method='spearman')
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'pearson' or 'spearman'")


def summarize_predixcan_results(
    results: List[Dict[str, Any]],
    region_metadata: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Summarize PrediXcan results across multiple regions.

    Args:
        results: List of result dictionaries from run_predixcan_for_region
        region_metadata: Optional DataFrame with region information

    Returns:
        DataFrame with summary statistics for each region

    Example:
        >>> summary = summarize_predixcan_results(all_results, peak_metadata)
        >>> print(summary[summary['test_pearson'] > 0.1])
    """
    summary_data = []

    for i, result in enumerate(results):
        row = {
            'region_idx': i,
            'status': result.get('status', 'unknown'),
            'n_variants': result.get('n_variants', 0),
            'n_active_variants': result.get('n_active_variants', 0),
            'optimal_alpha': result.get('optimal_alpha', np.nan),
            'train_r2': result.get('train_r2', np.nan),
            'train_pearson': result.get('train_pearson', np.nan),
            'val_r2': result.get('val_r2', np.nan),
            'val_pearson': result.get('val_pearson', np.nan),
            'test_r2': result.get('test_r2', np.nan),
            'test_pearson': result.get('test_pearson', np.nan)
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Merge with region metadata if provided
    if region_metadata is not None:
        summary_df = pd.concat([
            summary_df,
            region_metadata.reset_index(drop=True)
        ], axis=1)

    return summary_df


def _create_fallback_result(
    y_train: np.ndarray,
    y_val: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    status: str
) -> Dict[str, Any]:
    """
    Create fallback result for failed models.

    Returns predictions as the mean of training values.
    """
    train_mean = np.mean(y_train) if len(y_train) > 0 else 0.0

    return {
        'model': None,
        'status': status,
        'optimal_alpha': np.nan,
        'n_variants': 0,
        'n_active_variants': 0,
        'train_r2': np.nan,
        'train_pearson': np.nan,
        'val_r2': np.nan,
        'val_pearson': np.nan,
        'test_r2': np.nan,
        'test_pearson': np.nan,
        'train_pred': np.full(len(y_train), train_mean) if len(y_train) > 0 else np.array([]),
        'val_pred': np.full(len(y_val), train_mean) if y_val is not None and len(y_val) > 0 else None,
        'test_pred': np.full(len(y_test), train_mean) if y_test is not None and len(y_test) > 0 else None,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def calculate_additional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate additional regression metrics beyond R² and Pearson.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, and explained variance

    Example:
        >>> extra_metrics = calculate_additional_metrics(y_test, test_pred)
        >>> print(f"RMSE: {extra_metrics['rmse']:.3f}")
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
    }