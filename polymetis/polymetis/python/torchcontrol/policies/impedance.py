# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch

import torchcontrol as toco
from torchcontrol.transform import Transformation as T
from torchcontrol.transform import Rotation as R
from torchcontrol.utils import to_tensor, diagonalize_gain


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


# Variable impedance control
class HybridJointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space, but with both fixed joint gains and adaptive operational space gains.
    """

    def __init__(
        self,
        joint_pos_current,
        Kq,
        Kqd,
        Kx,
        Kxd,
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
        # Register gains as parameters
        # self.register_parameter("Kq", torch.nn.Parameter(diagonalize_gain(to_tensor(Kq))))
        # self.register_parameter("Kqd", torch.nn.Parameter(diagonalize_gain(to_tensor(Kqd))))
        # self.register_parameter("Kx", torch.nn.Parameter(diagonalize_gain(to_tensor(Kx))))
        # self.register_parameter("Kxd", torch.nn.Parameter(diagonalize_gain(to_tensor(Kxd))))

        # store vectors as parameters (shape-safe updates)
        self.register_parameter("Kq_vec",  torch.nn.Parameter(to_tensor(Kq).reshape(-1)))
        self.register_parameter("Kqd_vec", torch.nn.Parameter(to_tensor(Kqd).reshape(-1)))
        self.register_parameter("Kx_vec",  torch.nn.Parameter(to_tensor(Kx).reshape(-1)))
        self.register_parameter("Kxd_vec", torch.nn.Parameter(to_tensor(Kxd).reshape(-1)))

        # targets for smoothing like dynamic_reconfigure
        self.register_parameter("Kx_tgt_vec",  torch.nn.Parameter(self._param_dict["Kx_vec"].clone()))
        self.register_parameter("Kxd_tgt_vec", torch.nn.Parameter(self._param_dict["Kxd_vec"].clone()))
        self.register_parameter("Kq_tgt_vec",  torch.nn.Parameter(self._param_dict["Kq_vec"].clone()))
        self.register_parameter("Kqd_tgt_vec", torch.nn.Parameter(self._param_dict["Kqd_vec"].clone()))
        self.beta = 0.1  # per-tick blend factor (~LPF)


        self.joint_pd = toco.modules.feedback.HybridJointSpacePD()

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

        # smooth targets into current (server-side LPF like ROS)
        with torch.no_grad():
            for name in ["Kx_vec", "Kxd_vec", "Kq_vec", "Kqd_vec"]:
                cur = self._param_dict[name]
                tgt = self._param_dict[name.replace("_vec", "_tgt_vec")]
                cur.data = (1 - self.beta) * cur.data + self.beta * tgt.data

        # diagonalize at runtime (cheap; 6/7 elems)
        Kx  = diagonalize_gain(self._param_dict["Kx_vec"])
        Kxd = diagonalize_gain(self._param_dict["Kxd_vec"])
        Kq  = diagonalize_gain(self._param_dict["Kq_vec"])
        Kqd = diagonalize_gain(self._param_dict["Kqd_vec"])

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
