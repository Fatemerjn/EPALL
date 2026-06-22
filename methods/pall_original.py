"""PALL-Original -- the PALL baseline (no overlap protection).

Forgetting resets the forget-task subnet and replays the retained tasks to
recover knowledge transfer, WITHOUT protecting the critical shared overlap. This
is the baseline the overlap-aware PALL-Modified is compared against.

All logic lives in ``PALLBase`` (``methods/pall_base.py``); this subclass only
fixes the variant and documents the method. Registered for ``--method
pall_original``.
"""
from .pall_base import PALLBase


class PALLOriginal(PALLBase):
    def __init__(self, args):
        super(PALLOriginal, self).__init__(args)
        self.method_variant = "original"
