"""Independent simulator truth checks; never used to choose actor targets."""

import math

import numpy as np


def audit_surfaces(result, root_position, root_rotation, truth):
    records = []
    heading_world = math.atan2(root_rotation[1, 0], root_rotation[0, 0])
    for surface in result.surfaces:
        centroid_w = root_rotation @ surface.centroid + root_position
        candidates = np.flatnonzero((truth[:, 0] <= centroid_w[0]) & (truth[:, 1] > centroid_w[0]))
        record = {
            "surface_id": surface.surface_id, "valid": surface.valid,
            "track_id": surface.track_id, "last_observed_time_s": surface.last_observed_time,
            "kind": surface.kind, "rejection": surface.rejection,
            "tilt_deg": surface.tilt_deg, "plane_rms_m": surface.rms,
            "observed_cells": int(surface.observed_mask.sum()),
            "safe_centers": int(surface.safe_center_mask.sum()),
            "height_error_m": None, "near_error_m": None, "far_error_m": None,
            "edge_uncertainty_m": None, "safe_center_truth_violations": None,
            "safe_center_height_violations": None,
            "min_truth_clearance_m": None,
        }
        if not len(candidates):
            # A candidate outside the audit's known terrain must not silently pass.
            record["safe_center_truth_violations"] = record["safe_centers"]
        if len(candidates):
            interval = truth[candidates[0]]
            record["height_error_m"] = float(abs(centroid_w[2] - interval[2]))
            centers = result.candidate_centers(surface.surface_id)
            rear = result.cfg.edge_margin if result.direction >= 0 else result.cfg.other_end_margin
            front = result.cfg.edge_margin if result.direction <= 0 else result.cfg.other_end_margin
            half_length = 0.5 * (result.cfg.foot_rear_extent + result.cfg.foot_front_extent)
            half_width = result.cfg.foot_half_width + result.cfg.lateral_margin
            corners = np.array([[-half_length-rear, -half_width], [-half_length-rear, half_width],
                                [half_length+front, -half_width], [half_length+front, half_width]])
            yaw = result.heading_rad or 0.0
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            corners_world = (centers[:, None] + corners @ rotation.T) @ root_rotation[:2, :2].T + root_position[:2]
            clearance = np.minimum(corners_world[..., 0].min(axis=1) - interval[0],
                                   interval[1] - corners_world[..., 0].max(axis=1))
            record["safe_center_truth_violations"] = int((clearance < -1.0e-6).sum())
            center_heights = surface.height_at(centers) + root_position[2]
            record["safe_center_height_violations"] = int((np.abs(center_heights-interval[2]) > 0.02).sum())
            if len(clearance):
                record["min_truth_clearance_m"] = float(clearance.min())
            uncertainties = []
            for index, (name, edge) in enumerate((("near", surface.near_edge), ("far", surface.far_edge))):
                if edge is None:
                    continue
                y = float(surface.centroid[1])
                x = edge.x_at(y)
                edge_w = root_rotation @ np.array([x, y, surface.height_at([x, y])]) + root_position
                record[f"{name}_error_m"] = float(abs(edge_w[0] - interval[index]))
                uncertainties.append(edge.uncertainty)
            if uncertainties:
                record["edge_uncertainty_m"] = max(uncertainties)
        records.append(record)
    heading_error = None if result.heading_rad is None else abs(math.degrees(
        math.atan2(math.sin(result.heading_rad + heading_world), math.cos(result.heading_rad + heading_world))))
    return records, heading_error


def observed_next_tread_centers(result, root_position, root_rotation, sole_positions_w):
    """Diagnostic only: use measured sole heights, not terrain truth or contact confirmation."""
    return observed_next_tread_coverage(result, root_position, root_rotation, sole_positions_w)["centers"]


def observed_next_tread_coverage(result, root_position, root_rotation, sole_positions_w,
                                lateral_adjustment_m=0.08):
    """Check lateral candidates for both feet, not IK reachability or dynamic stability."""
    coverage = {"centers": 0, "left_centers": 0, "right_centers": 0,
                "both_feet_same_surface": False, "lateral_adjustment_m": lateral_adjustment_m}
    if result.direction == 0:
        return coverage
    support_height = float(np.min(np.asarray(sole_positions_w)[:, 2]))
    yaw = result.heading_rad or 0.
    rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    feet = (np.asarray(sole_positions_w)[:, :2]-root_position[:2]) @ root_rotation[:2, :2] @ rotation.T
    eligible = []
    for surface in result.surfaces:
        height = float((root_rotation @ surface.centroid + root_position)[2])
        advance = result.direction * (height-support_height)
        if surface.valid and result.cfg.min_riser_height-0.02 <= advance <= result.cfg.max_riser_height+0.02:
            eligible.append((advance, surface))
    nearest = min((advance for advance, _ in eligible), default=np.inf)
    for advance, surface in eligible:
        if advance <= nearest+0.035:
            centers = result.candidate_centers(surface.surface_id) @ rotation.T
            coverage["centers"] += len(centers)
            left = centers[np.abs(centers[:, 1]-feet[0, 1]) <= lateral_adjustment_m]
            right = centers[np.abs(centers[:, 1]-feet[1, 1]) <= lateral_adjustment_m]
            coverage["left_centers"] += len(left)
            coverage["right_centers"] += len(right)
            if len(left) and len(right):
                aligned = np.abs(left[:, None, 0]-right[None, :, 0]) <= 0.08
                separated = left[:, None, 1]-right[None, :, 1] >= 0.16
                coverage["both_feet_same_surface"] |= bool((aligned & separated).any())
    return coverage
