"""
smx_research.config
===================
Phase 1 shim: re-exports config helpers from the legacy smx/ directory.
Will be replaced by a DatasetConfig dataclass in Phase 7.
"""
from config import (  # type: ignore[import]
    load_dataset_config,
    list_available_datasets,
    get_compatible_datasets,
)

__all__ = [
    "load_dataset_config",
    "list_available_datasets",
    "get_compatible_datasets",
]
