"""Basic tests for the alphagenome_eval package."""

import pytest


def test_import_package():
    """Test that the main package can be imported."""
    import alphagenome_eval
    assert hasattr(alphagenome_eval, "__version__")


def test_import_utils():
    """Test that utility modules can be imported."""
    from alphagenome_eval.utils import data
    from alphagenome_eval.utils import modeling
    from alphagenome_eval.utils import coordinates
    from alphagenome_eval.utils import scoring


def test_import_workflows():
    """Test that workflow modules can be imported."""
    from alphagenome_eval.workflows import inference
    from alphagenome_eval.workflows import predixcan


def test_import_personal_dataset():
    """Test that PersonalDataset can be imported."""
    from alphagenome_eval import PersonalDataset
