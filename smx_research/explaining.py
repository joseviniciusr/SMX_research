"""
smx_research.explaining
=======================
Phase 1 shim: re-exports the legacy explaining module so that existing
experiment scripts continue to work without modification.

After Phase 2, the core SMX algorithm (extract_spectral_zones,
ZoneAggregator, PredicateGenerator, etc.) lives in the ``smx`` package.
This shim will be removed once run_experiment.py is updated to use the
new ``smx`` API in Phase 6.
"""
import smx_research  # noqa: F401 — ensures sys.path is set up

from explaining import *  # type: ignore[import]  # noqa: F401, F403
