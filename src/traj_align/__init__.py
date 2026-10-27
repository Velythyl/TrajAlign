"""Standalone VLM-assisted floor/trajectory alignment, extracted from RAGMAP.

Detects structural planes in a map, asks a VLM which one is the floor, and
writes floor-aligned copies of the trajectory, observations, map points and
point cloud (z up, floor at z=0, longest wall along an axis, centered).
"""
from traj_align.common.types import PlaneDetectionConfig
from traj_align.floor import FloorAlignedArtifacts, FloorAlignment, align_floor_artifacts

__all__ = ["PlaneDetectionConfig", "FloorAlignedArtifacts", "FloorAlignment", "align_floor_artifacts"]
