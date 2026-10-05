"""Depth-based geometric perception used by locomotion policies."""

from .stair_mode_gate import StairMode, StairModeGate, StairModeGateCfg
from .stair_geometry import StairGeometryExtractor, StairGeometryResult, compensate_tread_history, draw_stair_geometry_debug

__all__ = [
    "StairGeometryExtractor",
    "StairGeometryResult",
    "compensate_tread_history",
    "StairMode",
    "StairModeGate",
    "StairModeGateCfg",
    "draw_stair_geometry_debug",
]
