"""Compatibility alias for checkpoints pickled with the old module name."""
from sci_defs_v3 import SCIDetectV3, StructuralConsistencyV3, register_sci

__all__ = ["StructuralConsistencyV3", "SCIDetectV3", "register_sci"]
