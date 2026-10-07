"""Visible, physically actuated stair teaching, not depth transfer or learned-policy playback."""

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from legged_lab.perception.mujoco_step_teacher import MujocoStepTeacher
from legged_lab.perception.stair_step_controller import StairStepController, StepControlCfg, StepPhase
from legged_lab.perception.tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor


ROOT = Path(__file__).resolve().parents[2]
TASK_GOAL = "both_full_soles_supported_on_same_observed_tread"


def observed_mask_rectangles(mask):
    """Partition observed cells into rectangles without filling holes or gaps."""
    active, rectangles = {}, []
    for row, cells in enumerate(np.asarray(mask, dtype=bool)):
        edges = np.flatnonzero(np.diff(np.r_[False, cells, False]))
        runs = {(int(start), int(end)) for start, end in edges.reshape(-1, 2)}
        for run in sorted(active.keys() - runs):
            rectangles.append((active.pop(run), row, *run))
        for run in runs - active.keys():
            active[run] = row
    rectangles.extend((start, len(mask), *run) for run, start in sorted(active.items()))
    return rectangles


def observed_tread_quads(lock):
    """World-space faces of the locked observed support mask, on its fitted plane."""
    geometry = lock.geometry
    surface = next(s for s in geometry.surfaces if s.surface_id == lock.surface_id)
    half = geometry.cfg.grid_size/2
    quads = []
    for row_start, row_end, col_start, col_end in observed_mask_rectangles(surface.observed_mask):
        low = geometry.grid_xy[row_start, col_start]-half
        high = geometry.grid_xy[row_end-1, col_end-1]+half
        xy = np.array([[low[0], low[1]], [high[0], low[1]],
                       [high[0], high[1]], [low[0], high[1]]])
        local = np.column_stack((xy, surface.height_at(xy)))
        quads.append(local @ lock.rotation.T+lock.root_position)
    return np.asarray(quads).reshape(-1, 4, 3)


def _right_triangles(vertices):
    """Split a triangle into right triangles for MuJoCo's triangle primitive."""
    a, b, c = np.asarray(vertices)
    normal = np.cross(b-a, c-a)
    normal /= np.linalg.norm(normal)
    # The longest edge keeps the altitude inside the triangle, even on a tilted plane.
    edges = ((a, b, c), (b, c, a), (c, a, b))
    start, end, apex = max(edges, key=lambda edge: np.linalg.norm(edge[1]-edge[0]))
    base = end-start
    foot = start+base*(np.dot(apex-start, base)/np.dot(base, base))
    for endpoint in (start, end):
        x, y = endpoint-foot, apex-foot
        if min(np.linalg.norm(x), np.linalg.norm(y)) < 1.e-10:
            continue
        if np.dot(np.cross(x, y), normal) < 0:
            x, y = y, x
        size = np.array([np.linalg.norm(x), np.linalg.norm(y), 0.])
        rotation = np.column_stack((x/size[0], y/size[1], normal))
        yield foot, size, rotation


def draw_target_region(scene, quads, lock=None, show_foot_targets=False):
    """Draw only observed support; optional foot markers are control references."""
    scene.ngeom = 0
    if lock is None:
        return
    surface = next(s for s in lock.geometry.surfaces if s.surface_id == lock.surface_id)
    normal = lock.rotation @ surface.normal
    normal /= np.linalg.norm(normal)
    for quad in quads:
        raised = quad+.003*normal
        for indices in ((0, 1, 2), (0, 2, 3)):
            for position, size, rotation in _right_triangles(raised[list(indices)]):
                if scene.ngeom >= scene.maxgeom:
                    return  # Under-display complex masks; never replace them with a bounding box.
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_TRIANGLE,
                                   size, position, rotation.ravel(), np.array([.12, .95, .30, .38]))
                scene.ngeom += 1
    if show_foot_targets:
        cosine, sine = np.cos(lock.heading), np.sin(lock.heading)
        rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        cfg = lock.geometry.cfg
        for target in lock.targets:
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_LINEBOX,
                               np.array([.5*(cfg.foot_front_extent+cfg.foot_rear_extent),
                                         cfg.foot_half_width, .002]),
                               target+.006*normal, rotation.ravel(), np.array([1., .7, .12, .9]))
            scene.ngeom += 1


def draw_motion_status(scene, sample):
    """Update live status through the scene, without blocking overlay requests.

    Repeated Handle.set_texts calls stall for about one second on this local
    MuJoCo viewer build. Keep its screen overlay static and animate this label
    alongside the rest of the user scene instead.
    """
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    # At the default 130-degree view this offset is screen-right of the torso,
    # clear of the head silhouette and the fixed top-left screen overlay.
    position = np.asarray(sample["root"])+[.50, .45, .10]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), position,
                       np.eye(3).ravel(), np.array([1., 1., 1., 1.]))
    left, right = sample["load_fraction"]
    geom.label = f"{sample['phase']}  L/R: {left:.2f}/{right:.2f} BW"
    scene.ngeom += 1


def build_model(height=.11, width=.32, direction=1, *, with_depth_camera=False):
    source = ROOT/"legged_lab/assets/elf3_lite/xml/elf3.xml"
    xml = ET.parse(source).getroot()
    for include in list(xml.findall("include")):
        xml.remove(include)
    xml.find("compiler").set("meshdir", str(source.parent.parent/"meshes"))
    ET.SubElement(xml, "option", timestep="0.0025", integrator="implicitfast", iterations="50")
    ET.SubElement(xml, "statistic", center="0.3 0 0.7", extent="1.8")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.4 0.4 0.4")
    world = xml.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="4 2 .05", rgba=".65 .68 .71 1",
                  friction=".8 .01 .001", condim="3")
    near = .22
    for level in range(2):
        top = (level+1)*height if direction > 0 else (1-level)*height
        half_height = max(top/2, .01)
        ET.SubElement(world, "geom", name=f"tread_{level}", type="box",
                      pos=f"{near+(level+.5)*width} 0 {top-half_height}", size=f"{width/2} .8 {half_height}",
                      rgba=".36 .60 .56 1" if level == 0 else ".49 .55 .68 1", friction=".8 .01 .001")
    if direction < 0:
        ET.SubElement(world, "geom", name="start_platform", type="box", pos=f"-.89 0 {height}", size=f"1.11 .8 {height}")
        xml.find("worldbody/body[@name='torso_link']").set("pos", "0 0 1.32")
    # Hardware has no pressure sensors; contact forces below are explicitly an oracle.
    sensor = xml.find("sensor")
    for item in list(sensor):
        if item.tag == "touch":
            sensor.remove(item)
    if with_depth_camera:
        from legged_lab.perception.mujoco_depth_geometry import add_depth_camera
        add_depth_camera(xml)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        value = (-.30 if "hip_y_joint" in name or "ankle_y_joint" in name else
                 .60 if "knee_y_joint" in name else .2 if "shoulder_y_joint" in name else
                 .15 if name == "l_shoulder_x_joint" else -.15 if name == "r_shoulder_x_joint" else
                 .60 if "elbow_y_joint" in name else 0.)
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    teacher = MujocoStepTeacher(model, data)
    m = teacher.reader.measurement(data)
    source_height = 0. if direction > 0 else 2*height
    data.qpos[2] += source_height-m.sole_positions[:, 2].mean()-.00015
    mujoco.mj_forward(model, data)
    return model, data


class TeachingEpisode:
    def __init__(self, height=.11, width=.32, direction=1, *, geometry_source="known"):
        if geometry_source not in ("known", "depth"):
            raise ValueError("Geometry source must be known or depth.")
        self.geometry_source = geometry_source
        self.model, self.data = build_model(height, width, direction, with_depth_camera=geometry_source == "depth")
        # The coordinated posture/trajectory is an ascent candidate. Descent
        # retains its previous control until its own motion quality is improved.
        self.teacher = MujocoStepTeacher(self.model, self.data, posture_control=direction > 0)
        # Stay well inside the teacher's validated CoP limits (8 cm / 1.8 cm).
        # The body need not reach the sole center before the rear foot can lift.
        self.controller = StairStepController(StepControlCfg(support_com_half_length_m=.045 if direction > 0 else 0.,
                                                             support_com_half_width_m=.010 if direction > 0 else 0.,
                                                             overlap_swing_lift=direction > 0))
        self.direction = direction
        self.world_points = None
        if geometry_source == "known":
            x, y = np.meshgrid(np.arange(-.32, .94, .003), np.arange(-.38, .38, .003))
            levels = np.where(x < .22, 0, np.where(x < .22+width, 1, 2))
            heights = levels*height if direction > 0 else (2-levels)*height
            heights = np.where(x >= .22+2*width, 0., heights)
            self.world_points = np.column_stack((x.ravel(), y.ravel(), heights.ravel()))
        self.extractor = TreadSurfaceExtractor(SurfaceValidationCfg(min_forward=-.35, max_forward=1.1,
                                                                   lateral_half_width=.60, grid_size=.01))
        self.depth_source = None
        if geometry_source == "depth":
            from legged_lab.perception.mujoco_depth_geometry import MujocoDepthGeometry
            self.depth_source = MujocoDepthGeometry(self.model, self.extractor.cfg)
        self.geometry, self.geometry_time = None, -1.
        self.dt = .01
        self.initial_root = self.measurement().root_position.copy()
        self.initial_feet = self.measurement().sole_positions.copy()
        self.samples = []
        self.trace = []
        self.target_region = None
        self.target_region_quads = np.empty((0, 4, 3))

    def measurement(self):
        generation = 0 if self.depth_source is None else self.depth_source.memory.generation
        m = self.teacher.reader.measurement(self.data, generation=generation, include_contact_truth=True,
                                          camera_timestamp=self.geometry_time)
        hips = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("l_hip_z_link", "r_hip_z_link")]
        m.hip_offsets = self.data.xpos[hips]-m.root_position
        m.com_offset = self.data.subtree_com[self.teacher.reader.root_id]-m.root_position
        return m

    def refresh_geometry(self, measurement=None, *, tracking_only=False):
        """Refresh the observation source without changing a locked target."""
        m = self.measurement() if measurement is None else measurement
        if self.depth_source is not None:
            self.geometry = self.depth_source.update(self.data, tracking_only=tracking_only)
            self.geometry_time = self.depth_source.last_frame_time
            m.camera_timestamp = self.geometry_time
            m.generation = self.depth_source.memory.generation
            return self.geometry
        if m.timestamp-self.geometry_time >= .08:
            self.geometry_time = m.timestamp
            self.geometry = self.extractor.extract((self.world_points-m.root_position) @ m.yaw_rotation)
            self.geometry.timestamp_s = m.timestamp
            for surface in self.geometry.surfaces:
                world_z = float(surface.height_at(surface.centroid[:2]))+m.root_position[2]
                surface.track_id = int(round(world_z/.01))
                surface.last_observed_time = m.timestamp
            m.camera_timestamp = self.geometry_time
        elif self.geometry is not None:
            # Reproject synthetic world truth at the control pose; no fake depth frame is created.
            self.geometry = self.extractor.extract((self.world_points-m.root_position) @ m.yaw_rotation)
            self.geometry.timestamp_s = self.geometry_time
            for surface in self.geometry.surfaces:
                surface.track_id = int(round((float(surface.height_at(surface.centroid[:2]))+m.root_position[2])/.01))
                surface.last_observed_time = self.geometry_time
        return self.geometry

    def close(self):
        if self.depth_source is not None:
            self.depth_source.close()

    def tick(self):
        mujoco.mj_forward(self.model, self.data)
        m, c = self.measurement(), self.controller
        state = (float(self.data.time), self.data.qpos.copy(), self.data.qvel.copy())
        self.refresh_geometry(m)
        c.update(self.geometry, m, self.dt)
        c.constrain_body_reference(m)
        if c.lock is not None and self.target_region is None:
            self.target_region_quads = observed_tread_quads(c.lock)
            vertices = self.target_region_quads.reshape(-1, 3)
            self.target_region = {
                "task_goal": TASK_GOAL,
                "reference_role": "teacher_control_waypoints_not_required_foot_centers",
                "region_source": ("locked_observed_mask_from_rendered_depth" if self.geometry_source == "depth"
                                  else "locked_observed_mask_from_known_simulation_treads"),
                "track_id": int(c.lock.track_id), "locked_at": float(m.timestamp),
                "observed_quads_world": self.target_region_quads.tolist(),
                "observed_bounds_world": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                "actual_soles_world_at_lock": m.sole_positions.tolist(),
                "reference_soles_world": c.lock.targets.tolist(),
                "reference_lateral_shift_body_m": ((c.lock.targets-m.sole_positions) @ m.yaw_rotation)[:, 1].tolist(),
            }
            print("[TARGET_REGION] "+json.dumps(self.target_region), flush=True)
        if c.phase == StepPhase.RECOVER:
            raise RuntimeError("Action teacher stopped: "+c.failure_reason)
        active = np.ones(2, dtype=bool)
        if c.swing_foot is not None:
            swing = c.swing_foot
            active[swing] = (c.phase in (StepPhase.LOWER_LEAD, StepPhase.LOWER_TRAIL)
                             and c._physical_support(m, swing, min_load=c.cfg.touchdown_contact_fraction,
                                                     record_plant=False))
        loads = np.array([.5, .5])
        if c.phase in (StepPhase.SHIFT_LEAD, StepPhase.LIFT_LEAD, StepPhase.LOWER_LEAD):
            loads[c.lead], loads[1-c.lead] = 0., 1.
        elif c.phase in (StepPhase.TRANSFER, StepPhase.SHIFT_TRAIL, StepPhase.LIFT_TRAIL, StepPhase.LOWER_TRAIL):
            loads[c.lead], loads[1-c.lead] = 1., 0.
        if c.phase in (StepPhase.LOWER_LEAD, StepPhase.LOWER_TRAIL):
            loads[c.swing_foot], loads[1-c.swing_foot] = .15, .85
        root = c.reference_root if c.active else self.initial_root
        feet = (c.reference_feet if c.active else self.initial_feet).copy()
        minimum_loads = np.zeros(2)
        for foot in range(2):
            if c.confirmed_plants[foot]:
                feet[foot, 2] -= c.cfg.touchdown_probe_depth
                minimum_loads[foot] = .15
        step_features = c.features(m)
        motor_torques = []
        for _ in range(4):
            self.data.ctrl[:] = self.teacher.control(root, feet, active, self.model.opt.timestep, loads,
                                                    minimum_loads)
            motor_torques.append(self.data.ctrl.copy())
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or self.data.qpos[2] < .65:
            raise RuntimeError("Action teacher lost physical balance.")
        self.trace.append((*state, step_features, np.stack(motor_torques)))
        sample = {"time": round(m.timestamp, 5), "next_time": round(float(self.data.time), 5), "phase": c.phase.name,
                  "root": m.root_position.tolist(), "feet": m.sole_positions.tolist(),
                  "load_fraction": (m.contact_forces[:, 2]/m.body_weight).tolist(),
                  "shift_conditions": {name: bool(value) for name, value in c.shift_requirements.items()},
                  "failure": c.failure_reason,
                  "success": c.phase == StepPhase.COMPLETE}
        self.samples.append(sample)
        return sample

    def save_trace(self, path):
        if not self.samples or not self.samples[-1]["success"]:
            raise ValueError("Only physically successful episodes may be saved as motion candidates.")
        np.savez_compressed(path,
                            state_time=np.array([row[0] for row in self.trace]),
                            qpos=np.stack([row[1] for row in self.trace]),
                            qvel=np.stack([row[2] for row in self.trace]),
                            step_features=np.stack([row[3] for row in self.trace]),
                            motor_torques=np.stack([row[4] for row in self.trace]),
                            actuator_joint_names=np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                                          int(joint)) for joint in self.teacher.joints]),
                            physics_dt=self.model.opt.timestep, supervisor_dt=self.dt,
                            task_goal=TASK_GOAL,
                            target_region_quads_world=self.target_region_quads,
                            reference_soles_world=self.controller.lock.targets,
                            reference_role="teacher_control_waypoints_not_required_foot_centers",
                            trace_role="physical_motion_candidate", motion_quality_validated=False,
                            controller_profile="coordinated_ascent" if self.direction > 0 else "legacy_descent",
                            geometry_source=self.geometry_source,
                            known_geometry=self.geometry_source == "known", force_oracle=True, learned_policy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--loop", action="store_true", help="Replay episodes in the visible viewer until closed.")
    parser.add_argument("--direction", choices=("up", "down"), default="up")
    parser.add_argument("--controller", choices=("dynamic", "staged"), default="dynamic",
                        help="Dynamic up/down motion candidate, or the earlier staged baseline.")
    parser.add_argument("--geometry_source", choices=("known", "depth"), default="known",
                        help="Use known treads or an actual rendered D435i depth stream for target locking.")
    parser.add_argument("--initial_forward_offset", type=float, default=.05,
                        help="Dynamic scene initial forward offset in metres; applied before physics, not an approach skill.")
    parser.add_argument("--show_foot_targets", action="store_true",
                        help="Show optional teacher foot reference markers inside the shared tread region.")
    parser.add_argument("--output_dir", default="logs/mujoco_stair_teacher")
    args = parser.parse_args()
    if args.duration <= 0 or (args.headless and args.loop):
        raise ValueError("Positive duration required; --loop requires a visible viewer.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    direction = 1 if args.direction == "up" else -1
    if args.controller == "dynamic":
        from legged_lab.perception.mujoco_stair_motion import DynamicTeachingEpisode
        episode = DynamicTeachingEpisode(direction=direction, initial_forward_offset=args.initial_forward_offset,
                                         geometry_source=args.geometry_source)
    else:
        episode = TeachingEpisode(direction=direction, geometry_source=args.geometry_source)
    viewer = None if args.headless else mujoco.viewer.launch_passive(episode.model, episode.data)
    if viewer is not None:
        viewer.cam.lookat[:] = [.3, 0., .65]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 130., -18.
        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          f"{args.direction.upper()} STAIRS | ACTION TEACHER (not learned policy)",
                          "green: observed target tread"))
    print(f"[MODE] PHYSICAL ACTION TEACHER | geometry={args.geometry_source} | NOT learned-policy playback", flush=True)
    print("[PROFILE] "+json.dumps({"controller": args.controller,
                                   "initial_forward_offset_m": getattr(episode, "initial_forward_offset", 0.)}), flush=True)
    index, failure, last_phase = 0, "", ""
    try:
        while viewer is None or viewer.is_running():
            started = time.monotonic()
            try:
                sample = episode.tick()
            except RuntimeError as exc:
                failure = str(exc)
                break
            if sample["phase"] != last_phase or len(episode.samples) % 100 == 0:
                print("[TEACHER] "+json.dumps(sample), flush=True)
                last_phase = sample["phase"]
            if viewer is not None:
                with viewer.lock():
                    draw_target_region(viewer.user_scn, episode.target_region_quads,
                                       episode.controller.lock, args.show_foot_targets)
                    draw_motion_status(viewer.user_scn, sample)
                viewer.sync()
                time.sleep(max(0., episode.dt-(time.monotonic()-started)))
            done = sample["phase"] in ("COMPLETE", "RECOVER") or episode.data.time >= args.duration
            if done:
                report = {"mode": "physical_action_teacher", "geometry_source": args.geometry_source,
                          "known_geometry": args.geometry_source == "known",
                          "learned_policy": False, "depth_transfer_verified": False, "force_oracle": True,
                          "root_fixed": False, "teleport_after_reset": False, "success": sample["success"],
                          "motion_quality_validated": False,
                          "controller_profile": getattr(episode, "profile", "coordinated_ascent" if episode.direction > 0 else "legacy_descent"),
                          "initial_forward_offset_m": getattr(episode, "initial_forward_offset", 0.),
                          "failure": sample["failure"], "samples": episode.samples,
                          "task_goal": TASK_GOAL, "target_region": episode.target_region,
                          "show_foot_targets": args.show_foot_targets,
                          "max_planned_normalized_dynamics_residual": episode.teacher.max_dynamics_residual}
                record_episode = not args.loop or index == 0
                if report["success"] and record_episode:
                    trace_path = output/f"{args.direction}_{index:03d}_candidate.npz"
                    episode.save_trace(trace_path)
                    report["candidate_trace"] = str(trace_path)
                if record_episode:
                    (output/f"{args.direction}_{index:03d}.json").write_text(json.dumps(report, indent=2)+"\n")
                print("[RESULT] "+json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
                if not args.loop:
                    break
                time.sleep(1.)
                if args.controller == "dynamic":
                    episode = episode.reset()
                    last_phase, index = "", index+1
                    continue
                mujoco.mj_resetData(episode.model, episode.data)
                episode.close()
                replacement = TeachingEpisode(direction=episode.direction, geometry_source=args.geometry_source)
                episode.data.qpos[:] = replacement.data.qpos
                mujoco.mj_forward(episode.model, episode.data)
                replacement.model, replacement.data = episode.model, episode.data
                if replacement.depth_source is not None:
                    replacement.depth_source.model = episode.model
                replacement.teacher = MujocoStepTeacher(episode.model, episode.data,
                                                       posture_control=episode.direction > 0)
                episode, last_phase, index = replacement, "", index+1
        if failure:
            (output/f"{args.direction}_{index:03d}_failed.json").write_text(json.dumps(
                {"mode": "physical_action_teacher", "success": False, "failure": failure,
                 "geometry_source": args.geometry_source, "known_geometry": args.geometry_source == "known",
                 "task_goal": TASK_GOAL, "target_region": episode.target_region,
                 "samples": episode.samples}, indent=2)+"\n")
            if viewer is not None:
                print("[FAILED] "+failure+"; viewer retained until closed.", flush=True)
                viewer.set_texts((None, None, "ACTION TEACHER: FAILED (simulation stopped)", failure))
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.03)
            raise RuntimeError(failure)
    finally:
        episode.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
