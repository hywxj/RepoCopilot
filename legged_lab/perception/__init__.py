"""Depth-based geometric perception used by locomotion policies."""

from importlib import import_module


# Keep NumPy/MuJoCo helpers usable without importing the torch policy stack.
_EXPORT_MODULES = {
    "StairGeometryExtractor": ".stair_geometry",
    "StairGeometryResult": ".stair_geometry",
    "compensate_tread_history": ".stair_geometry",
    "StairMode": ".stair_mode_gate",
    "StairModeGate": ".stair_mode_gate",
    "StairModeGateCfg": ".stair_mode_gate",
    "draw_stair_geometry_debug": ".stair_geometry",
}

__all__ = [
    "StairGeometryExtractor",
    "StairGeometryResult",
    "compensate_tread_history",
    "StairMode",
    "StairModeGate",
    "StairModeGateCfg",
    "draw_stair_geometry_debug",
]


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
