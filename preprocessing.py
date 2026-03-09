"""
smx_research.preprocessing
===========================
Phase 1 shim: re-exports preprocessing functions from the legacy smx/
directory. Will be replaced by proper fit/transform classes in Phase 3.
"""
from preprocessings import (  # type: ignore[import]
    poisson,
    modified_poisson,
    pareto,
    mc,
    auto_scaling,
    msc,
)

__all__ = [
    "poisson",
    "modified_poisson",
    "pareto",
    "mc",
    "auto_scaling",
    "msc",
]
