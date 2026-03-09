"""
smx_research — Phase 1 compatibility shim
==========================================
This package temporarily re-exports from the legacy ``smx/`` directory
(the monolithic research code at the repository root).

As phases 3-6 of the refactoring plan are completed, each sub-module
will be replaced with a proper implementation, and this shim will be
removed. Do not add new functionality here — add it directly as a class
in the appropriate ``smx_research`` sub-package.

Installation (development mode)
--------------------------------
    pip install -e ./SMX   # installs the smx core library
    pip install -e .       # installs this package (smx_research)
"""
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Legacy import bridge: add the old smx/ directory to sys.path so the
# shim sub-modules (preprocessing, models, etc.) can import from it.
# This sys.path manipulation is intentionally kept INSIDE the package so
# that user-facing files (e.g. experiments/run_experiment.py) remain clean.
# ---------------------------------------------------------------------------
_LEGACY_DIR = Path(__file__).resolve().parent.parent / "smx"
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

from smx_research._version import __version__  # noqa: F401

__all__ = ["__version__"]
