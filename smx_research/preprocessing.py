"""
smx_research.preprocessing
===========================
Phase 1 shim: re-exports preprocessing functions from the legacy smx/
directory. Will be replaced by proper fit/transform classes in Phase 3.
"""
# smx_research.__init__ already added the legacy smx/ dir to sys.path
import smx_research  # noqa: F401 — ensures sys.path is set up

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
