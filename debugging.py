"""
smx_research.debugging
======================
Phase 1 shim: re-exports debugging/comparison utilities from the legacy
smx/ directory. Will be moved to smx_research/_contrib/ in Phase 5.
"""
import smx_research  # noqa: F401 — ensures sys.path is set up

from debugging import *  # type: ignore[import]  # noqa: F401, F403
