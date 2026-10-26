import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
nb_path = ROOT / "kaggle-train-rtdetr-peking_used.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Preserve the training function code (from CLASS_NAMES onward) from the existing Cell 5
old_c5 = nb["cells"][5]["source"]
# Find the CLASS_NAMES line to preserve the training function exactly
split_marker = '\nCLASS_NAMES = globals().get("CLASS_NAMES"'
if split_marker not in old_c5:
    split_marker = "\nCLASS_NAMES = globals().get('CLASS_NAMES'"
marker_idx = old_c5.find(split_marker)
if marker_idx == -1:
    raise ValueError("Cannot find CLASS_NAMES marker in Cell 5")

training_func_code = old_c5[marker_idx:]  # from CLASS_NAMES to end

NEW_C5_PREAMBLE = """# Cell 5: Standardized Training Pipeline for Peking University RT-DETR Models
import sys, types, enum
import numpy as np
import torch
import torch._utils
import importlib.machinery
import torch.nn.init as _init

# === PATCH ORDER IS CRITICAL - DO NOT REORDER ===

# STEP 0: Giải phóng modules rt_detr bị cache lỗi
for mod in list(sys.modules.keys()):
    if mod.startswith("transformers.models.rt_detr") or mod.startswith("transformers.image_"):
        del sys.modules[mod]

# STEP 1: Patch _chunk_or_narrow_cat NGAY ĐẦU TIÊN
# (phải có TRƯỚC KHI import accelerate/transformers/torchvision)
if not hasattr(torch._utils, "_chunk_or_narrow_cat"):
    def _chunk_or_narrow_cat(tensor, num_chunks, narrow_dim, cat_dim=0):
        return torch.cat(torch.chunk(tensor, num_chunks, dim=narrow_dim), dim=cat_dim)
    torch._utils._chunk_or_narrow_cat = _chunk_or_narrow_cat

# STEP 2: Remove broken torchvision (C++ NMS crash)
for k in ["torchvision", "torchvision.transforms", "torchvision.transforms.functional"]:
    sys.modules.pop(k, None)

# STEP 3: Patch transformers import_utils (torchvision / distributed / flex_attn / tensor stubs)
try:
    import transformers.utils.import_utils as _t_iu
    _orig_pkg = getattr(_t_iu, "_is_package_available", None)
    if _orig_pkg is not None:
        def _safe_pkg_avail(pkg_name, return_version=False):
            if pkg_name == "torchvision":
                return (False, None) if return_version else False
            return _orig_pkg(pkg_name, return_version)
        _t_iu._is_package_available = _safe_pkg_avail
    _t_iu.is_torchvision_available = lambda: False
    _t_iu._torchvision_available = False
    _t_iu.is_torch_distributed_available = lambda: False
    _t_iu.is_torch_flex_attn_available = lambda: False

    import transformers.utils as _t_u
    _t_u.is_torchvision_available = lambda: False
    _t_u._torchvision_available = False
    _t_u.is_torch_distributed_available = lambda: False
    _t_u.is_torch_flex_attn_available = lambda: False
    for _attr in ["is_jax_tensor", "is_flax_tensor", "is_tf_tensor"]:
        if not hasattr(_t_u, _attr):
            setattr(_t_u, _attr, lambda x: False)
    if not hasattr(_t_u, "is_torch_tensor"):
        setattr(_t_u, "is_torch_tensor", lambda x: isinstance(x, torch.Tensor))
    try:
        import transformers.utils.generic as _t_ug
        for _attr in ["is_jax_tensor", "is_flax_tensor", "is_tf_tensor"]:
            if not hasattr(_t_ug, _attr):
                setattr(_t_ug, _attr, lambda x: False)
        if not hasattr(_t_ug, "is_torch_tensor"):
            setattr(_t_ug, "is_torch_tensor", lambda x: isinstance(x, torch.Tensor))
    except Exception:
        pass
except Exception:
    pass

# STEP 4: Mock torch._dynamo (Bypass FlexAttention & Dynamo for Python 3.13)
class _MockDynamoModule(types.ModuleType):
    __path__ = []
    def __getattr__(self, name):
        if name == "config":
            m = types.ModuleType("torch._dynamo.config")
            m.__spec__ = importlib.machinery.ModuleSpec("torch._dynamo.config", None)
            return m
        def _dummy(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda fn: fn
        return _dummy

_mock_dynamo = _MockDynamoModule("torch._dynamo")
_mock_dynamo.__path__ = []
_mock_dynamo.__spec__ = importlib.machinery.ModuleSpec("torch._dynamo", None, is_package=True)
_mock_dynamo.assume_constant_result = lambda fn: fn
_mock_dynamo.disable = lambda fn=None, *args, **kwargs: (lambda f: f) if fn is None else fn
_mock_dynamo.is_compiling = lambda: False
_mock_dynamo.config = types.ModuleType("torch._dynamo.config")
_mock_dynamo.config.__spec__ = importlib.machinery.ModuleSpec("torch._dynamo.config", None)

sys.modules["torch._dynamo"] = _mock_dynamo
sys.modules["torch._dynamo.config"] = _mock_dynamo.config
torch._dynamo = _mock_dynamo

for _submod in ["_trace_wrapped_higher_order_op", "backends", "utils", "variables"]:
    _full = f"torch._dynamo.{_submod}"
    if _full not in sys.modules:
        _m = types.ModuleType(_full)
        _m.__spec__ = importlib.machinery.ModuleSpec(_full, None)
        _m.TransformGetItemToIndex = type("TransformGetItemToIndex", (), {})
        sys.modules[_full] = _m

# eval_frame mock (needed by accelerate.is_compiled_module → OptimizedModule)
_eval_frame = types.ModuleType("torch._dynamo.eval_frame")
_eval_frame.__spec__ = importlib.machinery.ModuleSpec("torch._dynamo.eval_frame", None)
_eval_frame.OptimizedModule = type("OptimizedModule", (types.ModuleType,), {})
sys.modules["torch._dynamo.eval_frame"] = _eval_frame
_mock_dynamo.eval_frame = _eval_frame

# STEP 5: Fix torch.distributed.tensor → PROPER PACKAGE with ALL submodules
# CRITICAL: must have __path__ = [] so Python treats it as a package,
# otherwise "torch.distributed.tensor._utils" raises "is not a package"
_mock_dt = types.ModuleType("torch.distributed.tensor")
_mock_dt.__path__ = []
_mock_dt.__spec__ = importlib.machinery.ModuleSpec("torch.distributed.tensor", None, is_package=True)
_mock_dt.DTensor = None
_mock_dt.DeviceMesh = None
_mock_dt.init_device_mesh = None
sys.modules["torch.distributed.tensor"] = _mock_dt

for _sub_name, _sub_attrs in [
    ("_dtensor_spec", {"DTensorSpec": None, "TensorMeta": None}),
    ("_utils", {"compute_local_shape_and_global_offset": lambda *a, **kw: (None, None)}),
    ("placement_types", {
        "_StridedShard": type("_StridedShard", (), {}),
        "Shard": type("Shard", (), {}),
        "Replicate": type("Replicate", (), {}),
        "Partial": type("Partial", (), {}),
    }),
]:
    _sub_full = f"torch.distributed.tensor.{_sub_name}"
    _sub_mod = types.ModuleType(_sub_full)
    _sub_mod.__spec__ = importlib.machinery.ModuleSpec(_sub_full, None)
    for _k, _v in _sub_attrs.items():
        setattr(_sub_mod, _k, _v)
    sys.modules[_sub_full] = _sub_mod

# STEP 6: Mock torch.distributed.checkpoint (prevents FakeStore/FSDP ImportError)
_mock_dc = types.ModuleType("torch.distributed.checkpoint")
_mock_dc.__path__ = []
_mock_dc.__spec__ = importlib.machinery.ModuleSpec("torch.distributed.checkpoint", None, is_package=True)
sys.modules["torch.distributed.checkpoint"] = _mock_dc

for _dc_sub, _dc_attrs in [
    ("api", {"CheckpointException": type("CheckpointException", (Exception,), {})}),
    ("default_planner", {
        "DefaultLoadPlanner": type("DefaultLoadPlanner", (), {}),
        "DefaultSavePlanner": type("DefaultSavePlanner", (), {}),
    }),
    ("filesystem", {
        "FileSystemReader": type("FileSystemReader", (), {}),
        "FileSystemWriter": type("FileSystemWriter", (), {}),
    }),
    ("hf_storage", {
        "HuggingFaceStorageReader": type("HuggingFaceStorageReader", (), {}),
        "HuggingFaceStorageWriter": type("HuggingFaceStorageWriter", (), {}),
    }),
    ("planner", {
        "LoadItemType": type("LoadItemType", (), {}),
        "WriteItemType": type("WriteItemType", (), {}),
        "SavePlan": type("SavePlan", (), {}),
        "LoadPlan": type("LoadPlan", (), {}),
    }),
    ("metadata", {"Metadata": type("Metadata", (), {})}),
]:
    _sub_full = f"torch.distributed.checkpoint.{_dc_sub}"
    _sub_mod = types.ModuleType(_sub_full)
    _sub_mod.__spec__ = importlib.machinery.ModuleSpec(_sub_full, None)
    for _k, _v in _dc_attrs.items():
        setattr(_sub_mod, _k, _v)
    sys.modules[_sub_full] = _sub_mod

# Patch transformers distributed checkpoint availability check
try:
    import transformers.distributed.checkpoint as _tdc
    _tdc._check_distributed_checkpointing_available = lambda: False
except Exception:
    pass

# STEP 7: Patch accelerate (now _chunk_or_narrow_cat is available from STEP 1)
try:
    import accelerate.utils.other as _acc_other
    import transformers.modeling_utils as _tm
    _acc_other.is_compiled_module = lambda module: False
    _acc_other.has_compiled_regions = lambda module: False
    _acc_other.is_torch_distributed_available = lambda: False
    def _identity_extract(model, **kwargs): return model
    _acc_other.extract_model_from_parallel = _identity_extract
    _tm.extract_model_from_parallel = _identity_extract
    print("✅ accelerate/transformers save_pretrained stubs: OK")
except Exception as _e:
    print(f"⚠️  accelerate patch warning: {_e}")

# STEP 8: Patch torch.utils._config_module.Config (deprecated kwarg)
try:
    import torch.utils._config_module as _cm
    _old_config = _cm.Config
    def _safe_config(*args, **kwargs):
        kwargs.pop("deprecated", None)
        kwargs.pop("deprecation_message", None)
        return _old_config(*args, **kwargs)
    _cm.Config = _safe_config
except Exception:
    pass

# STEP 9: Patch torch.compiler.disable (no-op decorator)
try:
    import torch.compiler
    torch.compiler.disable = lambda fn=None, *args, **kwargs: (lambda f: f) if fn is None else fn
except Exception:
    pass

# STEP 10: Patch torch._library.opaque_object (is_custom_class_obj)
try:
    import torch._library.opaque_object as _oo
    if not hasattr(_oo, "is_custom_class_obj"):
        _oo.is_custom_class_obj = lambda x: False
except Exception:
    if "torch._library.opaque_object" not in sys.modules:
        _oo = types.ModuleType("torch._library.opaque_object")
        _oo.__spec__ = importlib.machinery.ModuleSpec("torch._library.opaque_object", None)
        _oo.is_custom_class_obj = lambda x: False
        sys.modules["torch._library.opaque_object"] = _oo

# STEP 11: AggressiveGuardFreeMode mock
class AggressiveGuardFreeMode(enum.IntEnum):
    DISABLED = 0
    ALWAYS_GUARD = 0
    GUARD_OR_TRUE = 1
    GUARD_OR_FALSE = 2

try:
    import torch.fx.experimental._config as _fx_cfg
    _cls = type(_fx_cfg)
    _orig_getattr = getattr(_cls, "__getattr__", None)
    def _safe_getattr(self, name):
        if name == "AggressiveGuardFreeMode":
            return AggressiveGuardFreeMode
        if _orig_getattr is not None:
            return _orig_getattr(self, name)
        raise AttributeError(f"{name} does not exist")
    _cls.__getattr__ = _safe_getattr
    if "torch.fx.experimental._config" in sys.modules:
        sys.modules["torch.fx.experimental._config"].__dict__["AggressiveGuardFreeMode"] = AggressiveGuardFreeMode
except Exception:
    pass

# STEP 12: symbolic_shapes patches (has_hint, size_hint, etc.)
try:
    import torch.fx.experimental.symbolic_shapes as _sym_shapes
    _sym_shapes.has_hint = lambda a: True
    _sym_shapes.size_hint = lambda a: int(a) if hasattr(a, "__int__") else 1
    _sym_shapes.guard_scalar = lambda a: a
    _sym_shapes.guard_size_oblivious = lambda a: a
    _sym_shapes.is_symbolic = lambda a: False
except Exception:
    pass

# STEP 13: Bypass torch._refs in nn.init via NumPy C-buffers
def _safe_normal_(tensor, mean=0.0, std=1.0, generator=None):
    with torch.no_grad():
        arr = np.random.normal(mean, std, size=tensor.shape).astype(np.float32)
        tensor.copy_(torch.from_numpy(arr))
        return tensor

def _safe_uniform_(tensor, a=0.0, b=1.0, generator=None):
    with torch.no_grad():
        arr = np.random.uniform(a, b, size=tensor.shape).astype(np.float32)
        tensor.copy_(torch.from_numpy(arr))
        return tensor

_init.normal_ = _safe_normal_
_init._no_grad_normal_ = _safe_normal_
_init.uniform_ = _safe_uniform_
_init._no_grad_uniform_ = _safe_uniform_

# STEP 14: Import RTDetrForObjectDetection
try:
    from transformers import RTDetrForObjectDetection, RTDetrConfig
    print("✅ ĐÃ NẠP THÀNH CÔNG RTDetrForObjectDetection!")
except Exception as e:
    print(f"Fallback import attempt due to: {e}")
    from transformers.models.rt_detr.modeling_rt_detr import RTDetrForObjectDetection, RTDetrConfig
import torch.optim as optim"""

new_c5 = NEW_C5_PREAMBLE + training_func_code

nb["cells"][5]["source"] = new_c5

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"SUCCESS: Rewrote Cell 5 with correct patch ordering ({len(new_c5.splitlines())} lines)")
print(f"Saved: {nb_path}")

