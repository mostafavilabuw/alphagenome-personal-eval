import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def save_alphagenome_predictions(
    predictions: Dict[str, Any], 
    output_dir: str = "alphagenome_outputs",
    filename_prefix: str = "predictions",
    save_format: str = "npz",
    include_metadata: bool = True,
    compress: bool = True
) -> Dict[str, str]:
    """
    Unified function to save AlphaGenome prediction outputs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    if save_format == "npz":
        arrays_to_save = {}
        metadata = {}
        for pred_name, pred_output in predictions.items():
            for attr_name in dir(pred_output):
                if not attr_name.startswith('_'):
                    try:
                        attr_obj = getattr(pred_output, attr_name)
                        if hasattr(attr_obj, 'values'):
                            array_name = f"{pred_name}_{attr_name}"
                            arrays_to_save[array_name] = attr_obj.values
                            if include_metadata:
                                metadata[array_name] = {
                                    'shape': attr_obj.values.shape,
                                    'dtype': str(attr_obj.values.dtype),
                                    'prediction_type': pred_name,
                                    'output_type': attr_name
                                }
                                if hasattr(attr_obj, 'interval'):
                                    interval = attr_obj.interval
                                    metadata[array_name]['interval'] = {
                                        'chromosome': getattr(interval, 'chromosome', None),
                                        'start': getattr(interval, 'start', None),
                                        'end': getattr(interval, 'end', None),
                                        'length': len(interval) if hasattr(interval, '__len__') else None
                                    }
                    except (AttributeError, TypeError):
                        continue
        save_func = np.savez_compressed if compress else np.savez
        arrays_file = output_path / f"{filename_prefix}_{timestamp}.npz"
        save_func(arrays_file, **arrays_to_save)
        saved_files['arrays'] = str(arrays_file)
        if include_metadata and metadata:
            metadata_file = output_path / f"{filename_prefix}_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_file)
    elif save_format == "npy":
        for pred_name, pred_output in predictions.items():
            for attr_name in dir(pred_output):
                if not attr_name.startswith('_'):
                    try:
                        attr_obj = getattr(pred_output, attr_name)
                        if hasattr(attr_obj, 'values'):
                            array_file = output_path / f"{filename_prefix}_{pred_name}_{attr_name}_{timestamp}.npy"
                            np.save(array_file, attr_obj.values)
                            saved_files[f"{pred_name}_{attr_name}"] = str(array_file)
                    except (AttributeError, TypeError):
                        continue
    elif save_format == "pickle":
        pickle_file = output_path / f"{filename_prefix}_complete_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(predictions, f)
        saved_files['complete_data'] = str(pickle_file)
    else:
        raise ValueError(f"Unsupported save_format: {save_format}. Use 'npz', 'npy', or 'pickle'")
    return saved_files

def load_alphagenome_predictions(file_path: str, format_type: str = "auto") -> Dict[str, Any]:
    """
    Load saved AlphaGenome predictions.
    """
    file_path = Path(file_path)
    if format_type == "auto":
        format_type = file_path.suffix[1:]
    if format_type == "npz":
        return dict(np.load(file_path, allow_pickle=True))
    elif format_type == "npy":
        return np.load(file_path, allow_pickle=True)
    elif format_type in ["pkl", "pickle"]:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format_type: {format_type}") 