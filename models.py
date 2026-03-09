"""
smx_research.models
===================
Phase 1 shim: re-exports model training functions from the legacy smx/
directory. Will be replaced by proper BaseSpectralModel classes in Phase 4.
"""
from modeling import (  # type: ignore[import]
    pls_optimized,
    svm_optimized,
    mlp_optimized,
    vip_scores,
    explained_variance_from_scores,
)

__all__ = [
    "pls_optimized",
    "svm_optimized",
    "mlp_optimized",
    "vip_scores",
    "explained_variance_from_scores",
]
