"""PALL-Modified -- the MAIN overlap-aware selective-forgetting method.

Identifies the critical shared overlap between the forget task and the retained
tasks and protects it during forgetting. The critical subset is ranked by
``--protect_importance``:

  * ``conflict`` : relu(-g_forget * g_retain) on the rehearsal buffer -- targets
    the parameters where forgetting and retention directly compete; best when the
    forget/retain parameter overlap is HIGH.
  * ``gradient`` : |grad L_retain| on the rehearsal buffer (default).
  * ``weight``   : |w| absolute weight magnitude (legacy ablation).

All logic lives in ``PALLBase`` (``methods/pall_base.py``); this subclass only
fixes the variant and documents the method. Registered for ``--method
pall_modified`` (and the deprecated alias ``--method pall``).
"""
from .pall_base import PALLBase


class PALLModified(PALLBase):
    def __init__(self, args):
        super(PALLModified, self).__init__(args)
        # Defensive: the variant is normally set by normalize_method(); pin it
        # here too so the class is correct even if instantiated directly.
        self.method_variant = "modified"
