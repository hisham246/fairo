import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

urdf_path = "/home/robohub/polymetis/polymetis/data/franka_panda/panda_arm_tcp.urdf"
ee_name = "panda_hand_tcp"

robot_pin = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[])
model = robot_pin.model

print("nq:", model.nq, "nv:", model.nv)
print("num joints:", model.njoints)

# Pinocchio stores joints by name; links are usually "frames".
# For a URDF link, it typically appears as a frame:
frame_id = model.getFrameId(ee_name)
print("frame_id:", frame_id)

# Also useful: list a few last frames to sanity-check
for i in range(model.nframes-10, model.nframes):
    f = model.frames[i]
    print(i, f.name, "parent joint:", f.parent)
