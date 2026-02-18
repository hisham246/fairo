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


rm = robot.robot_model

for attr in ["get_link_names", "get_frame_names", "link_names", "frame_names"]:
    print(attr, "exists:", hasattr(rm, attr))

if hasattr(rm, "get_link_names"):
    names = rm.get_link_names()
    print("num links:", len(names))
    print("ee idx:", names.index(robot.metadata.ee_link_name))
elif hasattr(rm, "get_frame_names"):
    names = rm.get_frame_names()
    print("num frames:", len(names))
    print("ee idx:", names.index(robot.metadata.ee_link_name))
else:
    print("This build doesn't expose link/frame names; use Method 3 below.")

