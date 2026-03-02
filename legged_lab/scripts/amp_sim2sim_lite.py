

import argparse
import os
import sys

import mujoco
import mujoco_viewer
import numpy as np
import torch
import onnxruntime as ort
from pynput import keyboard
import time

class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 200.0
        num_action = 29
        # num_obs_per_step = 102
        num_obs_per_step = 96
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        # clip_observations = 100.0
        # clip_actions = 100.0
        # action_scale = 0.25
        # action_scale = np.array([
        #     0.231, 0.154, 0.213,  # waist_y, waist_x, waist_z
        #     0.213, 0.213, 0.231,  # l_hip_y, l_hip_x, l_hip_z
        #     0.213, 0.373, 0.230,  # l_knee_y, l_ankle_y, l_ankle_x
        #     0.213, 0.213, 0.231,  # r_hip_y, r_hip_x, r_hip_z
        #     0.213, 0.373, 0.230,  # r_knee_y, r_ankle_y, r_ankle_x
        #     0.231, 0.231, 0.373,  # l_shoulder_y, l_shoulder_x, l_shoulder_z
        #     0.231, 0.373,         # l_elbow_y, l_wrist_x
        #     0.373, 0.373,         # l_wrist_y, l_wrist_z
        #     0.231, 0.231, 0.373,  # r_shoulder_y, r_shoulder_x, r_shoulder_z
        #     0.231, 0.373,         # r_elbow_y, r_wrist_x
        #     0.373, 0.373,         # r_wrist_y, r_wrist_z
        # ])
        action_scale = np.array([
            0.231, 0.231, 0.231,
            0.231, 0.231, 0.154,
            0.373, 0.373, 0.213,
            0.231, 0.231, 0.213,
            0.213, 0.373, 0.373,
            0.213, 0.213, 0.373, 
            0.373, 0.231,
            0.231, 0.373, 
            0.373, 0.213, 0.213, 
            0.373, 0.373, 
            0.23, 0.23,
        ])

    # class robot:
    #     gait_air_ratio_l: float = 0.38  #步态空中比率l
    #     gait_air_ratio_r: float = 0.38  #步态空中比率r
    #     gait_phase_offset_l: float = 0.38  #步态相位偏移l
    #     gait_phase_offset_r: float = 0.88  #步态相位偏移r
    #     gait_cycle: float = 0.85    #步态周期（秒）


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        # self.policy = torch.jit.load(network_path)
        self.policy = ort.InferenceSession(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        # self.default_dof_pos = np.array(
        #     [   # 指定的固定关节角度
        #     0.0, 0.0, 0.0,
        #     -0.4,0.0,0.0,0.8,-0.4,0.0,
        #     -0.4,0.0,0.0,0.8,-0.4,0.0,
        #     0.5,0.3,-0.1,-0.2, 0.0,0.0,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
        #     0.5,-0.3,0.1,-0.2, 0.0,0.0,0.0]
        # )
        self.default_dof_pos = np.array(
            [   # 指定的固定关节角度
            0.0, 0.0, 0.0,
            -0.3,0.0,0.0,0.6,-0.3,0.0,
            -0.3,0.0,0.0,0.6,-0.3,0.0,
            0.2,0.2,0.0,0.6, 0.0,0.0,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
            0.2,-0.2,0.0,0.6, 0.0,0.0,0.0]
        )
        # self.default_dof_pos = np.array(
        #     [   # 指定的固定关节角度
        #     0.0, 0.0, 0.0,
        #     -0.2,0.0,0.0,0.4,-0.2,0.0,
        #     -0.2,0.0,0.0,0.4,-0.2,0.0,
        #     0.3,0.2,0.0,0.87, 0.0,0.0,0.0,     # 左臂放在大腿旁边 (Y=0 肩平, X=0 前后居中, Z=0 不旋转, 肘关节微弯)
        #     0.3,-0.2,0.0,0.87, 0.0,0.0,0.0]
        # )
        # self.joint_kp = np.array([     # 奔跑的关节kp，和joint_name顺序一一对应
        #     300,300,300,
        #     150,100,100,200,50,20,
        #     150,100,100,200,50,20,
        #     80,80,80,60, 20,20,20,
        #     80,80,80,60, 20,20,20], 
        #     dtype=np.float32)
        
        # self.joint_kd = np.array([  # 奔跑的关节kd，和joint_name顺序一一对应
        #     3,3,3,
        #     2,2,2,2.5,1,1,
        #     2,2,2,2.5,1,1,
        #     2,2,2,2, 1,1,1,
        #     2,2,2,2, 1,1,1], 
        #     dtype=np.float32)
        
        self.joint_kp = np.array([     # 奔跑的关节kp，和joint_name顺序一一对应
            108.448,162.672,176.421,
            176.421,176.421,154.22400,176.421,33.493,21.771,
            176.421,176.421,154.22400,176.421,33.493,21.771,
            54.224,54.224,16.747,54.224, 16.747,16.747,16.747,
            54.224,54.224,16.747,54.224, 16.747,16.747,16.747,
            ], 
            dtype=np.float32)

        self.joint_kd = np.array([  # 奔跑的关节kd，和joint_name顺序一一对应
            6.904,10.356,11.231,
            11.231,11.231,3.452,11.231,2.132,1.386,
            11.231,11.231,3.452,11.231,2.132,1.386,
            3.452,3.452,1.066,3.452, 1.066,1.066,1.066,
            3.452,3.452,1.066,3.452, 1.066,1.066,1.066,
            ], 
            dtype=np.float32)

        self.episode_length_buf = 0
        # self.gait_phase = np.zeros(2) #步态相位
        # self.gait_cycle = self.cfg.robot.gait_cycle
        # self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r]) #步态空中比率[0.38,0.38]
        # self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r]) #步态相位偏移[0.38,0.88]

        self.mujoco_to_isaac_idx = [
            15,    # 'l_shoulder_y_joint', 0
            22,    #  'r_shoulder_y_joint', 1
            0,    #  'waist_y_joint', 2
            16,    #  'l_shoulder_x_joint',3 
            23,    #  'r_shoulder_x_joint', 4
            1,    #  'waist_x_joint', 5
            17,    #  'l_shoulder_z_joint',6 
            24,    #  'r_shoulder_z_joint', 7
            2,    #  'waist_z_joint', 8
            18,    #  'l_elbow_y_joint',9 
            25,    #  'r_elbow_y_joint', 10
            3,    #  'l_hip_y_joint', 11
            9,    #  'r_hip_y_joint', 12
            19,    #  'l_wrist_x_joint',13 
            26,    #  'r_wrist_x_joint', 14
            4,    #  'l_hip_x_joint', 15
            10,   #  'r_hip_x_joint', 16
            20,    #  'l_wrist_y_joint', 17
            27,    #  'r_wrist_y_joint', 18
            5,    #  'l_hip_z_joint', 19
            11,   #  'r_hip_z_joint', 20
            21,    #  'l_wrist_z_joint', 21 
            28,    #  'r_wrist_z_joint', 22
            6,    #  'l_knee_y_joint', 23
            12,   #  'r_knee_y_joint', 24
            7,    #  'l_ankle_y_joint', 25
            13,   #  'r_ankle_y_joint', 26
            8,    #  'l_ankle_x_joint', 27
            14,   #  'r_ankle_x_joint',28

        ]
        self.isaac_to_mujoco_idx = [
            2,    # "waist_y_joint",
            5,    # "waist_x_joint",
            8,    # "waist_z_joint",
                
            11,    # "l_hip_y_joint",   # 左腿_髋关节_z轴
            15,    # "l_hip_x_joint",   # 左腿_髋关节_x轴
            19,    # "l_hip_z_joint",   # 左腿_髋关节_y轴
            23,    # "l_knee_y_joint",   # 左腿_膝关节_y轴
            25,    # "l_ankle_y_joint",   # 左腿_踝关节_y轴
            27,    # "l_ankle_x_joint",   # 左腿_踝关节_x轴

            12,    # "r_hip_y_joint",   # 右腿_髋关节_z轴    
            16,    # "r_hip_x_joint",   # 右腿_髋关节_x轴
            20,    # "r_hip_z_joint",   # 右腿_髋关节_y轴
            24,    # "r_knee_y_joint",   # 右腿_膝关节_y轴
            26,    # "r_ankle_y_joint",   # 右腿_踝关节_y轴
            28,    # "r_ankle_x_joint",   # 右腿_踝关节_x轴
            0,    # "l_shoulder_y_joint",   # 左臂_肩关节_y轴
            3,    # "l_shoulder_x_joint",   # 左臂_肩关节_x轴
            6,    # "l_shoulder_z_joint",   # 左臂_肩关节_z轴
            9,    # "l_elbow_y_joint",   # 左臂_肘关节_y轴
            13,    # "l_wrist_x_joint",
            17,    # "l_wrist_y_joint",
            21,    # "l_wrist_z_joint",
                
            1,    # "r_shoulder_y_joint",   # 右臂_肩关节_y轴   
            4,    # "r_shoulder_x_joint",   # 右臂_肩关节_x轴
            7,    # "r_shoulder_z_joint",   # 右臂_肩关节_z轴
            10,    # "r_elbow_y_joint",    # 右臂_肘关节_y轴
            14,    # "r_wrist_x_joint",
            18,    # "r_wrist_y_joint",
            22,    # "r_wrist_z_joint",
        ]
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )# 观测历史750

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        # self.dof_pos = self.data.sensordata[0:20]
        self.dof_pos = self.data.qpos[7:7+self.cfg.sim.num_action]
        # self.dof_vel = self.data.sensordata[20:40]
        self.dof_vel = self.data.qvel[6:6+self.cfg.sim.num_action]
        self.quat = self.data.qpos[3:7]# mj 
        self.omega = self.data.qvel[3:6]

        obs = np.concatenate(
            [
                # self.data.sensor("angular-velocity").data.astype(np.double),  # 3
                self.omega.astype(np.double),  # 3
                self.quat_rotate_inverse(
                    # self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
                    self.data.sensor("Body_Quat").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
                ),  # 3
                self.command_vel,  # 3
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],  # 29
                # (self.dof_pos - self.default_dof_pos),  # 29
                self.dof_vel[self.mujoco_to_isaac_idx],  # 29
                # np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 29 #限制动作值
                self.action,  # 29 #限制动作值
                # np.sin(2 * np.pi * self.gait_phase),  # 2 #步态相位正弦值
                # np.cos(2 * np.pi * self.gait_phase),  # 2 #步态相位余弦值
                # self.phase_ratio,  # 2 #步态空中比率
            ],
            axis=0,
        ).astype(np.float32)

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step) #滚动历史观测
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy() #添加最新观测

        # return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations) #裁剪观测值100
        return self.obs_history #不裁剪观测值

    def position_control(self) -> np.ndarray:
        """
        Apply position control using scaled action.

        Returns:
            np.ndarray: Target joint positions in MuJoCo order.
        """
        actions_scaled = self.action * self.cfg.sim.action_scale
        return actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos
        # return actions_scaled + self.default_dof_pos
        # return self.default_dof_pos

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position commands"""
        return (target_q - q) * kp + (target_dq - dq) * kd

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()#启动keyboard监听器

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs() #100
            # self.action[:] = self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()[:29] #获取动作前29
            
            self.obs_tensor = np.expand_dims(self.obs_history, axis=0)
            self.action = self.policy.run(["actions"], {"obs": self.obs_tensor})[0][0]
            
            # print(len(self.obs_history))#750
            # print(len(self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()))#20
            # self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions) #clip 100 //20

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                tau = self.pd_control(
                    self.position_control(),
                    self.data.qpos[7:7+self.cfg.sim.num_action],
                    self.joint_kp,
                    np.zeros_like(self.joint_kp),
                    self.data.qvel[6:6+self.cfg.sim.num_action],
                    self.joint_kd,
                    # np.zeros_like(self.joint_kd),
                )
                self.data.ctrl = tau
                # self.data.ctrl = self.position_control()
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.episode_length_buf += 1
            # self.calculate_gait_para()

        self.listener.stop()
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    # def calculate_gait_para(self) -> None:
    #     """
    #     根据模拟时间和偏移更新步态相位参数。
    #     """
    #     t = self.episode_length_buf * self.dt / self.gait_cycle
    #     self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
    #     self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        # self.command_vel[idx] += increment
        # self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip
        if idx == 1:
            self.command_vel[idx] += increment
            self.command_vel[idx] = np.clip(self.command_vel[idx], -0.6, 0.6)  # y clip
            print(f"Adjusted command velocity: y={self.command_vel[1]:.2f}, y={self.command_vel[1]:.2f}")
        elif idx == 2:
            self.command_vel[idx] += increment
            self.command_vel[idx] = np.clip(self.command_vel[idx], -1.5, 1.5)  # yaw clip
            print(f"Adjusted command velocity: yaw={self.command_vel[2]:.2f}, y={self.command_vel[1]:.2f}")
        else:
            self.command_vel[idx] += increment
            # self.command_vel[idx] = np.clip(self.command_vel[idx], -0.5, 1.0)  # x clip
            self.command_vel[idx] = np.clip(self.command_vel[idx], -0.5, 3.0)  # x clip
            print(f"Adjusted command velocity: x={self.command_vel[0]:.2f}")

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8      x += 0.2
                    self.adjust_command_vel(0, 0.1)
                elif key.char == "5":  # NumPad 2      x -= 0.2
                    self.adjust_command_vel(0, -0.1)
                elif key.char == "6":  # NumPad 4      y -= 0.2
                    self.adjust_command_vel(1, -0.1)
                elif key.char == "4":  # NumPad 6      y += 0.2
                    self.adjust_command_vel(1, 0.1)
                elif key.char == "9":  # NumPad 7      yaw += 0.2
                    self.adjust_command_vel(2, -0.1)
                elif key.char == "7":  # NumPad 9      yaw -= 0.2
                    self.adjust_command_vel(2, 0.1)
                elif key.char == "1":  # NumPad 5      停止
                    self.command_vel[:] = 0.0
                    print("Command velocity reset to zero")
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--task",
        type=str,
        default="walk_elf3",
        choices=["walk_elf3", "run_elf3"],
        help="Task type: 'walk' or 'run' to set gait parameters",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt. If not specified, it will be set automatically based on --task",
    )
    parser.add_argument(
        "--model",
        type=str,
        # default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/tienkung2_lite/mjcf/tienkung.xml"),
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/elf3_lite/xml/elf3.xml"),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=1000.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    if args.policy is None:
        args.policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", f"{args.task}.pt")

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded task preset: {args.task.upper()}")
    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    # Set gait parameters according to task
    # if args.task == "walk":
    #     sim_cfg.robot.gait_air_ratio_l = 0.38
    #     sim_cfg.robot.gait_air_ratio_r = 0.38
    #     sim_cfg.robot.gait_phase_offset_l = 0.38
    #     sim_cfg.robot.gait_phase_offset_r = 0.88
    #     sim_cfg.robot.gait_cycle = 0.85
    # elif args.task == "run":
    #     sim_cfg.robot.gait_air_ratio_l = 0.6
    #     sim_cfg.robot.gait_air_ratio_r = 0.6
    #     sim_cfg.robot.gait_phase_offset_l = 0.6
    #     sim_cfg.robot.gait_phase_offset_r = 0.1
    #     sim_cfg.robot.gait_cycle = 0.5

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()
