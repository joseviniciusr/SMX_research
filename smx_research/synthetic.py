"""
smx_research.synthetic
======================
Phase 1 shim: re-exports synthetic data generation from the legacy smx/
directory. Will be replaced by a SyntheticSpectraGenerator class in Phase 7.
"""
import smx_research  # noqa: F401 — ensures sys.path is set up

from synthetic import (  # type: ignore[import]
    generate_synthetic_spectral_data,
)

__all__ = ["generate_synthetic_spectral_data"]
