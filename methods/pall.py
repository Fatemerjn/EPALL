"""Backward-compatibility shim.

The PALL implementation was split into:
  * ``methods/pall_base.py``     -> ``PALLBase`` (all the machinery)
  * ``methods/pall_modified.py`` -> ``PALLModified`` (main method)
  * ``methods/pall_original.py`` -> ``PALLOriginal`` (baseline)

Old code importing ``from methods.pall import PALL`` keeps working: ``PALL`` is an
alias for ``PALLBase``.
"""
from .pall_base import PALLBase
from .pall_modified import PALLModified
from .pall_original import PALLOriginal

PALL = PALLBase

__all__ = ["PALL", "PALLBase", "PALLModified", "PALLOriginal"]
