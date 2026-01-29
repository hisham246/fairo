# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch

import torchcontrol as toco
from torchcontrol.transform import Transformation as T
from torchcontrol.transform import Rotation as R
from torchcontrol.utils import to_tensor, diagonalize_gain
from typing import List, Dict, Optional
import numpy as np
import csv
from datetime import datetime
import pathlib

class JointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space.
    """

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


# # Fixed impedance control
# class HybridJointImpedanceControl(toco.PolicyModule):
#     """
#     Impedance control in joint space, but with both fixed joint gains and adaptive operational space gains.
#     """

#     def __init__(
#         self,
#         joint_pos_current,
#         Kq,
#         Kqd,
#         Kx,
#         Kxd,
#         robot_model: torch.nn.Module,
#         ignore_gravity=True):
#         """
#         Args:
#             joint_pos_current: Current joint positions
#             Kp: P gains in Cartesian space
#             Kd: D gains in Cartesian space
#             robot_model: A robot model from torchcontrol.models
#             ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
#         """
#         super().__init__()

#         # Initialize modules
#         self.robot_model = robot_model
#         self.invdyn = toco.modules.feedforward.InverseDynamics(
#             self.robot_model, ignore_gravity=ignore_gravity
#         )
#         self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

#         # For recording Cartesian force feedback
#         self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kx, Kxd)
#         # Reference pose
#         joint_pos_current = to_tensor(joint_pos_current)
#         ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
#             joint_pos_current
#         )
#         self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
#         self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
#         self.ee_vel_desired = torch.nn.Parameter(torch.zeros(3))
#         self.ee_rvel_desired = torch.nn.Parameter(torch.zeros(3))


#         # Reference pose
#         self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
#         self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)


#     def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """
#         Args:
#             state_dict: A dictionary containing robot states

#         Returns:
#             A dictionary containing the controller output
#         """
#         # State extraction
#         joint_pos_current = state_dict["joint_positions"]
#         joint_vel_current = state_dict["joint_velocities"]

#         # For recording Cartesian force feedback
#         jacobian = self.robot_model.compute_jacobian(joint_pos_current)
#         ee_twist_current = jacobian @ joint_vel_current
#         ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
#             joint_pos_current
#         )

#         wrench_feedback = self.pose_pd(
#             ee_pos_current,
#             ee_quat_current,
#             ee_twist_current,
#             self.ee_pos_desired,
#             self.ee_quat_desired,
#             torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
#         )

#         # Detach the tensor from the computation graph and move it to the CPU
#         wrench_feedback_tensor = wrench_feedback.detach().cpu()
#         wrench_feedback_list: List[float] = wrench_feedback_tensor.tolist()

#         # print("Cartesian force feedback:", wrench_feedback_list)

#         # Control logic
#         torque_feedback = self.joint_pd(
#             joint_pos_current,
#             joint_vel_current,
#             self.joint_pos_desired,
#             self.joint_vel_desired,
#             self.robot_model.compute_jacobian(joint_pos_current),
#         )
#         torque_feedforward = self.invdyn(
#             joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
#         )  # coriolis

#         torque_out = torque_feedback + torque_feedforward
        
#         # Gravity compensation only
#         # torque_out = torch.zeros_like(torque_feedforward)

#         return {"joint_torques": torque_out}    


class TorqueRateLimiter(torch.nn.Module):
    def __init__(self, delta_tau_max=1.0, n=7):
        super().__init__()
        self.register_buffer("prev_tau", torch.zeros(n))
        self.delta_tau_max = float(delta_tau_max)
    def forward(self, tau):
        dtau = torch.clamp(
            tau - self.prev_tau,
            -self.delta_tau_max,
            self.delta_tau_max,
        )
        out = self.prev_tau + dtau
        self.prev_tau = out.detach()
        return out
    
# Variable impedance control
class HybridJointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space, but with both fixed joint gains and adaptive operational space gains.
    """

    def __init__(self, joint_pos_current, Kq, Kqd, Kx, Kxd, robot_model, ignore_gravity=True,
                 log_wrench: bool = True,
                 log_dir: str = "./force_logs",
                 log_flush_every: int = 1):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )

        # For recording Cartesian force feedback
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kx, Kxd)
        # Reference pose
        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
        self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
        self.ee_vel_desired = torch.nn.Parameter(torch.zeros(3))
        self.ee_rvel_desired = torch.nn.Parameter(torch.zeros(3))

        # Register gains as parameters
        # self.register_parameter("Kq", torch.nn.Parameter(diagonalize_gain(to_tensor(Kq))))
        # self.register_parameter("Kqd", torch.nn.Parameter(diagonalize_gain(to_tensor(Kqd))))
        # self.register_parameter("Kx", torch.nn.Parameter(diagonalize_gain(to_tensor(Kx))))
        # self.register_parameter("Kxd", torch.nn.Parameter(diagonalize_gain(to_tensor(Kxd))))

        # self.tau_limiter = TorqueRateLimiter(delta_tau_max=1.0, n=self.joint_pos_desired.numel())

        # store vectors as parameters (shape-safe updates)
        # gains as vectors
        Kq_init  = to_tensor(Kq).reshape(-1)
        Kqd_init = to_tensor(Kqd).reshape(-1)
        Kx_init  = to_tensor(Kx).reshape(-1)
        Kxd_init = to_tensor(Kxd).reshape(-1)

        # current (smoothed) values
        self.Kq_vec  = torch.nn.Parameter(Kq_init.clone())
        self.Kqd_vec = torch.nn.Parameter(Kqd_init.clone())
        self.Kx_vec  = torch.nn.Parameter(Kx_init.clone())
        self.Kxd_vec = torch.nn.Parameter(Kxd_init.clone())

        # target values (what update_current_policy writes to)
        self.Kq_tgt_vec  = torch.nn.Parameter(Kq_init.clone())
        self.Kqd_tgt_vec = torch.nn.Parameter(Kqd_init.clone())
        self.Kx_tgt_vec  = torch.nn.Parameter(Kx_init.clone())
        self.Kxd_tgt_vec = torch.nn.Parameter(Kxd_init.clone())

        # smoothing factor
        self.beta: float = 0.1

        self.joint_pd = toco.modules.feedback.HybridJointSpacePD()

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

        # -------------------------------
        # Wrench CSV logging setup
        # -------------------------------
        self.log_wrench = log_wrench
        self._log_flush_every = int(log_flush_every)
        self._log_counter = 0

        self._csv_file = None
        self._csv_writer = None

        if self.log_wrench:
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._csv_path = os.path.join(log_dir, f"wrench_{ts}.csv")

            # newline="" avoids blank lines on Windows
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)

            # Header: wall-clock + unix time + wrench components
            self._csv_writer.writerow([
                "datetime_local", "unix_time_s",
                "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"
            ])
            self._csv_file.flush()

    def _smooth_gains(self):
            b = self.beta

            # in-place on parameters (JIT-friendly)
            self.Kq_vec.mul_(1.0 - b)
            self.Kq_vec.add_(b * self.Kq_tgt_vec)

            self.Kqd_vec.mul_(1.0 - b)
            self.Kqd_vec.add_(b * self.Kqd_tgt_vec)

            self.Kx_vec.mul_(1.0 - b)
            self.Kx_vec.add_(b * self.Kx_tgt_vec)

            self.Kxd_vec.mul_(1.0 - b)
            self.Kxd_vec.add_(b * self.Kxd_tgt_vec)

    def _log_wrench_row(self, wrench_list: List[float]) -> None:
        if (not self.log_wrench) or (self._csv_writer is None):
            return

        now = datetime.now()
        row = [
            now.isoformat(timespec="microseconds"),
            now.timestamp(),
            *wrench_list[:6],  # expects 6D wrench; safe if list is longer
        ]
        self._csv_writer.writerow(row)

        self._log_counter += 1
        if self._log_flush_every > 0 and (self._log_counter % self._log_flush_every == 0):
            self._csv_file.flush()

    def close(self):
        # call this when shutting down to ensure file is closed properly
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
            finally:
                self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """

        # smooth gains (no .data, no _param_dict)
        self._smooth_gains()

        # diagonalize at runtime
        Kx  = diagonalize_gain(self.Kx_vec)
        Kxd = diagonalize_gain(self.Kxd_vec)
        Kq  = diagonalize_gain(self.Kq_vec)
        Kqd = diagonalize_gain(self.Kqd_vec)
        
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        # torque_feedback = self.joint_pd(
        #     joint_pos_current,
        #     joint_vel_current,
        #     self.joint_pos_desired,
        #     self.joint_vel_desired,
        #     self.robot_model.compute_jacobian(joint_pos_current),
        #     self._param_dict["Kq"],
        #     self._param_dict["Kqd"],
        #     self._param_dict["Kx"],
        #     self._param_dict["Kxd"],
        # )

        # print("Stiffness:", Kx.diag())
        # print("Damping:", Kxd.diag())

        # For recording Cartesian force feedback
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_twist_current = jacobian @ joint_vel_current
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )

        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            self.ee_pos_desired,
            self.ee_quat_desired,
            torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
        )

        # Detach the tensor from the computation graph and move it to the CPU
        wrench_feedback_tensor = wrench_feedback.detach().cpu()
        wrench_feedback_list: List[float] = wrench_feedback_tensor.tolist()

        self._log_wrench_row(wrench_feedback_list)

        # print("Cartesian force feedback:", wrench_feedback_list)

        torque_feedback = self.joint_pd(
            joint_pos_current, joint_vel_current,
            self.joint_pos_desired, self.joint_vel_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
            Kq, Kqd, Kx, Kxd
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}

class CartesianImpedanceControl(toco.PolicyModule):
    """
    Performs impedance control in Cartesian space.
    Errors and feedback are computed in Cartesian space, and the resulting forces are projected back into joint space.
    """

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kp, Kd)

        # Reference pose
        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
        self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
        self.ee_vel_desired = torch.nn.Parameter(torch.zeros(3))
        self.ee_rvel_desired = torch.nn.Parameter(torch.zeros(3))


    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_twist_current = jacobian @ joint_vel_current

        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            self.ee_pos_desired,
            self.ee_quat_desired,
            torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
        )
        torque_feedback = jacobian.T @ wrench_feedback

        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis

        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}
