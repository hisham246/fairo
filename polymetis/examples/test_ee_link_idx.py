import torch
import torchcontrol as toco
from polymetis import RobotInterface

robot = RobotInterface(ip_address="localhost")
q = robot.get_joint_positions()

# Use the exact URDF the server is serving (guaranteed to match what Polymetis uses)
import tempfile
urdf_text = robot.metadata.urdf_file
with tempfile.NamedTemporaryFile("w+", suffix=".urdf") as f:
    f.write(urdf_text)
    f.flush()

    rm_link8 = toco.models.RobotModelPinocchio(f.name, "panda_link8")
    rm_tcp   = toco.models.RobotModelPinocchio(f.name, "panda_hand_tcp")

    p8, q8 = rm_link8.forward_kinematics(q)
    pt, qt = rm_tcp.forward_kinematics(q)

    dp = (pt - p8)
    print("delta (tcp - link8):", dp.tolist(), "norm:", float(torch.linalg.norm(dp)))


pos_meta, quat_meta = robot.get_ee_pose()
pt, qt = rm_tcp.forward_kinematics(q)
print("pos diff norm:", float(torch.linalg.norm(pos_meta-pt)))
# quat diff check can be via dot product abs close to 1
print("quat dot abs:", float(torch.abs(torch.dot(quat_meta, qt))))
