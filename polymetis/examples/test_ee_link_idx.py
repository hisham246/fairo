import torch
from polymetis import RobotInterface

robot = RobotInterface(ip_address="localhost")

print("server ee_link_name:", robot.metadata.ee_link_name)

q = robot.get_joint_positions()

# FK through robot_model (this uses the EE link configured in RobotModelPinocchio)
pos, quat = robot.robot_model.forward_kinematics(q)
J = robot.robot_model.compute_jacobian(q)

print("FK pos:", pos.tolist())
print("FK quat:", quat.tolist())
print("Jacobian shape:", tuple(J.shape))   # should be (6, 7)
print("Jacobian finite:", torch.isfinite(J).all().item())
