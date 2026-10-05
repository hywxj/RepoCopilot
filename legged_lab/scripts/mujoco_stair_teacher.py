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
from legged_lab.perception.stair_step_controller import StairStepController, StepPhase
from legged_lab.perception.tread_surfaces import SurfaceValidationCfg, TreadSurfaceExtractor


ROOT = Path(__file__).resolve().parents[2]


def build_model(height=.11, width=.32, direction=1):
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
    def __init__(self, height=.11, width=.32, direction=1):
        self.model, self.data = build_model(height, width, direction)
        self.teacher = MujocoStepTeacher(self.model, self.data)
        self.controller = StairStepController()
        self.direction = direction
        x, y = np.meshgrid(np.arange(-.32, .94, .003), np.arange(-.38, .38, .003))
        levels = np.where(x < .22, 0, np.where(x < .22+width, 1, 2))
        heights = levels*height if direction > 0 else (2-levels)*height
        heights = np.where(x >= .22+2*width, 0., heights)
        self.world_points = np.column_stack((x.ravel(), y.ravel(), heights.ravel()))
        self.extractor = TreadSurfaceExtractor(SurfaceValidationCfg(min_forward=-.35, max_forward=1.1,
                                                                   lateral_half_width=.60, grid_size=.01))
        self.geometry, self.geometry_time = None, -1.
        self.dt = .01
        self.initial_root = self.measurement().root_position.copy()
        self.initial_feet = self.measurement().sole_positions.copy()
        self.samples = []
        self.trace = []

    def measurement(self):
        m = self.teacher.reader.measurement(self.data, include_contact_truth=True, camera_timestamp=self.geometry_time)
        hips = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("l_hip_z_link", "r_hip_z_link")]
        m.hip_offsets = self.data.xpos[hips]-m.root_position
        m.com_offset = self.data.subtree_com[self.teacher.reader.root_id]-m.root_position
        return m

    def tick(self):
        mujoco.mj_forward(self.model, self.data)
        m, c = self.measurement(), self.controller
        state = (float(self.data.time), self.data.qpos.copy(), self.data.qvel.copy())
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
        c.update(self.geometry, m, self.dt)
        c.constrain_body_reference(m)
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
            raise ValueError("Only physically successful episodes may become expert demonstrations.")
        np.savez_compressed(path,
                            state_time=np.array([row[0] for row in self.trace]),
                            qpos=np.stack([row[1] for row in self.trace]),
                            qvel=np.stack([row[2] for row in self.trace]),
                            step_features=np.stack([row[3] for row in self.trace]),
                            motor_torques=np.stack([row[4] for row in self.trace]),
                            actuator_joint_names=np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                                          int(joint)) for joint in self.teacher.joints]),
                            physics_dt=self.model.opt.timestep, supervisor_dt=self.dt,
                            known_geometry=True, force_oracle=True, learned_policy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--loop", action="store_true", help="Replay episodes in the visible viewer until closed.")
    parser.add_argument("--direction", choices=("up", "down"), default="up")
    parser.add_argument("--output_dir", default="logs/mujoco_stair_teacher")
    args = parser.parse_args()
    if args.duration <= 0 or (args.headless and args.loop):
        raise ValueError("Positive duration required; --loop requires a visible viewer.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode = TeachingEpisode(direction=1 if args.direction == "up" else -1)
    viewer = None if args.headless else mujoco.viewer.launch_passive(episode.model, episode.data)
    if viewer is not None:
        viewer.cam.lookat[:] = [.3, 0., .65]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 130., -18.
    print("[MODE] PHYSICAL ACTION TEACHER | known simulation treads | NOT learned-policy/depth-transfer playback", flush=True)
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
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "ACTION TEACHER (not learned policy)\n"+sample["phase"],
                                  f"known simulation treads\nload L/R: {sample['load_fraction'][0]:.2f} / {sample['load_fraction'][1]:.2f}"))
                with viewer.lock():
                    targets = episode.controller.lock
                    viewer.user_scn.ngeom = 0 if targets is None else 2
                    if targets is not None:
                        for foot in range(2):
                            mujoco.mjv_initGeom(viewer.user_scn.geoms[foot], mujoco.mjtGeom.mjGEOM_BOX,
                                               np.array([.12, .042, .003]), targets.targets[foot]+[0., 0., .003],
                                               np.eye(3).ravel(), np.array([.2, .9, .25, .3]))
                viewer.sync()
                time.sleep(max(0., episode.dt-(time.monotonic()-started)))
            done = sample["phase"] in ("COMPLETE", "RECOVER") or episode.data.time >= args.duration
            if done:
                report = {"mode": "physical_action_teacher", "geometry_source": "known_simulation_treads",
                          "learned_policy": False, "depth_transfer_verified": False, "force_oracle": True,
                          "root_fixed": False, "teleport_after_reset": False, "success": sample["success"],
                          "failure": sample["failure"], "samples": episode.samples,
                          "max_planned_normalized_dynamics_residual": episode.teacher.max_dynamics_residual}
                record_episode = not args.loop or index == 0
                if report["success"] and record_episode:
                    trace_path = output/f"{args.direction}_{index:03d}_expert.npz"
                    episode.save_trace(trace_path)
                    report["expert_trace"] = str(trace_path)
                if record_episode:
                    (output/f"{args.direction}_{index:03d}.json").write_text(json.dumps(report, indent=2)+"\n")
                print("[RESULT] "+json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
                if not args.loop:
                    break
                time.sleep(1.)
                mujoco.mj_resetData(episode.model, episode.data)
                replacement = TeachingEpisode(direction=episode.direction)
                episode.data.qpos[:] = replacement.data.qpos
                mujoco.mj_forward(episode.model, episode.data)
                replacement.model, replacement.data = episode.model, episode.data
                replacement.teacher = MujocoStepTeacher(episode.model, episode.data)
                episode, last_phase, index = replacement, "", index+1
        if failure:
            (output/f"{args.direction}_{index:03d}_failed.json").write_text(json.dumps(
                {"mode": "physical_action_teacher", "success": False, "failure": failure,
                 "samples": episode.samples}, indent=2)+"\n")
            if viewer is not None:
                print("[FAILED] "+failure+"; viewer retained until closed.", flush=True)
                viewer.set_texts((None, None, "ACTION TEACHER: FAILED (simulation stopped)", failure))
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.03)
            raise RuntimeError(failure)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
