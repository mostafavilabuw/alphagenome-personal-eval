"""
Utility modules for AlphaGenome evaluation.

This package provides reusable functions organized by functionality:
- data: Loading peaks, genes, VCF files, and creating datasets
- genome: Variant extraction, MAF calculation, personal sequence generation
- prediction: AlphaGenome model initialization and prediction wrappers
- modeling: ElasticNet training, PrediXcan pipeline, evaluation metrics
- visualization: Plotting functions for results analysis
"""

# Import key functions from each module for convenient access
from .data import (
    load_peaks,
    load_genes,
    load_vcf_samples,
    load_predixcan_results,
    create_genome_dataset,
    align_samples
)

from .genome import (
    extract_variants_in_region,
    calculate_maf,
    filter_variants_by_maf,
    get_personal_sequences,
    count_variants_in_regions,
    summarize_variants,
    VariantInfo
)

from .prediction import (
    init_dna_model,
    predict_atac,
    predict_rna,
    predict_personal_genome_atac,
    predict_personal_genome_rna,
    aggregate_over_region,
    batch_predict_regions,
    calculate_prediction_statistics
)

from .borzoi_utils import (
    # Constants
    BORZOI_SEQ_LEN,
    BORZOI_BIN_SIZE,
    BORZOI_NUM_BINS,
    BORZOI_NUM_TRACKS,
    BORZOI_PRED_LENGTH,
    BORZOI_SUPPORTED_CONTEXT_WINDOWS,
    # Model name mappings (for Borzoi and Flashzoi)
    BORZOI_MODEL_NAMES,
    FLASHZOI_MODEL_NAMES,
    DEFAULT_BORZOI_MODEL,
    DEFAULT_FLASHZOI_MODEL,
    # Core functions
    validate_borzoi_context_window,
    get_model_name,
    init_borzoi_model,
    dna_to_onehot,
    predict_borzoi,
    get_borzoi_track_indices,
    get_borzoi_dnase_track_indices,
    get_tissue_track_dict,
    # Aggregation
    aggregate_borzoi_over_region,
    # Personal genome prediction - RNA
    predict_borzoi_personal_genome,
    predict_borzoi_rna,
    predict_borzoi_personal_rna,
    # ATAC-seq / DNase chromatin accessibility
    predict_borzoi_atac,
    predict_borzoi_personal_atac,
    predict_borzoi_atac_multi_tissue,
    predict_borzoi_personal_atac_multi_tissue,
    # RNA paired strand prediction (for ROSMAP brain)
    get_borzoi_rna_track_by_identifier,
    predict_borzoi_personal_rna_paired_strand,
    # Multi-track CSV support
    load_tracks_from_csv,
    predict_borzoi_multi_track,
)

from .modeling import (
    train_elasticnet,
    evaluate_predictions,
    run_predixcan_for_region,
    calculate_correlation_matrix,
    summarize_predixcan_results,
    calculate_additional_metrics
)

from .visualization import (
    plot_prediction_vs_observed,
    plot_correlation_heatmap,
    plot_distribution,
    plot_correlation_distributions,
    plot_train_vs_test_comparison,
    plot_performance_vs_variants,
    plot_boxplot_comparison,
    create_comprehensive_predixcan_plots
)

from .gpu import (
    get_best_gpu,
    is_cuda_available,
    select_device,
    get_device_info
)

from .scoring import (
    VariantScorer,
    AGGREGATION_METHODS,
)

from .binning import (
    # Constants
    ALPHAGENOME_RESOLUTION,
    BORZOI_RESOLUTION,
    DEFAULT_BIN_SIZE,
    DEFAULT_WINDOW_SIZE,
    # Core binning functions
    bin_atac_track,
    bin_atac_around_tss,
    # Feature extraction
    extract_binned_features_borzoi,
    extract_binned_features_alphagenome,
    create_feature_matrix_for_gene,
    # Prediction saving/loading
    save_all_predictions,
    load_all_predictions,
)

__all__ = [
    # Data loading
    'load_peaks',
    'load_genes',
    'load_vcf_samples',
    'load_predixcan_results',
    'create_genome_dataset',
    'align_samples',

    # Genome operations
    'extract_variants_in_region',
    'calculate_maf',
    'filter_variants_by_maf',
    'get_personal_sequences',
    'count_variants_in_regions',
    'summarize_variants',
    'VariantInfo',

    # AlphaGenome Predictions
    'init_dna_model',
    'predict_atac',
    'predict_rna',
    'predict_personal_genome_atac',
    'predict_personal_genome_rna',
    'aggregate_over_region',
    'batch_predict_regions',
    'calculate_prediction_statistics',
    
    # Borzoi constants
    'BORZOI_SEQ_LEN',
    'BORZOI_BIN_SIZE',
    'BORZOI_NUM_BINS',
    'BORZOI_NUM_TRACKS',
    'BORZOI_PRED_LENGTH',
    'BORZOI_SUPPORTED_CONTEXT_WINDOWS',
    # Borzoi/Flashzoi model name mappings
    'BORZOI_MODEL_NAMES',
    'FLASHZOI_MODEL_NAMES',
    'DEFAULT_BORZOI_MODEL',
    'DEFAULT_FLASHZOI_MODEL',
    # Borzoi core functions
    'validate_borzoi_context_window',
    'get_model_name',
    'init_borzoi_model',
    'dna_to_onehot',
    'predict_borzoi',
    'get_borzoi_track_indices',
    'get_borzoi_dnase_track_indices',
    'get_tissue_track_dict',
    # Borzoi aggregation
    'aggregate_borzoi_over_region',
    # Borzoi personal genome - RNA
    'predict_borzoi_personal_genome',
    'predict_borzoi_rna',
    'predict_borzoi_personal_rna',
    # Borzoi ATAC-seq / DNase chromatin accessibility
    'predict_borzoi_atac',
    'predict_borzoi_personal_atac',
    'predict_borzoi_atac_multi_tissue',
    'predict_borzoi_personal_atac_multi_tissue',
    # Borzoi RNA paired strand prediction (for ROSMAP brain)
    'get_borzoi_rna_track_by_identifier',
    'predict_borzoi_personal_rna_paired_strand',
    # Borzoi multi-track CSV support
    'load_tracks_from_csv',
    'predict_borzoi_multi_track',

    # Modeling
    'train_elasticnet',
    'evaluate_predictions',
    'run_predixcan_for_region',
    'calculate_correlation_matrix',
    'summarize_predixcan_results',
    'calculate_additional_metrics',

    # Visualization
    'plot_prediction_vs_observed',
    'plot_correlation_heatmap',
    'plot_distribution',
    'plot_correlation_distributions',
    'plot_train_vs_test_comparison',
    'plot_performance_vs_variants',
    'plot_boxplot_comparison',
    'create_comprehensive_predixcan_plots',

    # GPU utilities
    'get_best_gpu',
    'is_cuda_available',
    'select_device',
    'get_device_info',

    # Variant scoring
    'VariantScorer',
    'AGGREGATION_METHODS',

    # Binning utilities
    'ALPHAGENOME_RESOLUTION',
    'BORZOI_RESOLUTION',
    'DEFAULT_BIN_SIZE',
    'DEFAULT_WINDOW_SIZE',
    'bin_atac_track',
    'bin_atac_around_tss',
    'extract_binned_features_borzoi',
    'extract_binned_features_alphagenome',
    'create_feature_matrix_for_gene',
    # Prediction saving/loading
    'save_all_predictions',
    'load_all_predictions',
]