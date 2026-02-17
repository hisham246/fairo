# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import time
import torchcontrol as toco


from polymetis import RobotInterface

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    # robot.go_home()

    # Joint impedance control
    joint_positions = robot.get_joint_positions()

    # Cartesian impedance control
    print("Performing Cartesian impedance control...")
    ee_pos, ee_quat = robot.get_ee_pose()

    robot.start_cartesian_impedance()

    print("ee_link_name:", robot.metadata.ee_link_name)
    print("hz:", robot.metadata.hz)

    urdf_text = robot.metadata.urdf_file
    print("URDF length:", len(urdf_text))

    urdf_path = "data/franka_panda/panda_arm_tcp.urdf"
    rm = toco.models.RobotModelPinocchio(urdf_path, "panda_hand_tcp")

    # Depending on torchcontrol version, one of these exists:
    print("ee_link_name:", "panda_hand_tcp")
    print("model links:", rm.get_link_names() if hasattr(rm, "get_link_names") else "no get_link_names()")


    # with open("polymetis_urdf_from_server.urdf", "w") as f:
    #     f.write(urdf_text)

    # print("Saved to polymetis_urdf_from_server.urdf")


    # print("Performing joint impedance control...")
    # robot.start_joint_impedance()

    # for i in range(40):
    #     joint_positions += torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, -0.015, 0.0])
    #     robot.update_desired_joint_positions(joint_positions)
    #     time.sleep(0.1)

    # robot.terminate_current_policy()


    # for i in range(40):
    #     ee_pos += torch.Tensor([-0.0025, 0.0, 0.0])
    #     robot.update_desired_ee_pose(position=ee_pos)
    #     time.sleep(0.1)

    # robot.terminate_current_policy()