import torch
import csv
import time
from polymetis import RobotInterface
from torchcontrol.policies.impedance import HybridJointImpedanceControl 
from polymetis_pb2 import Empty

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(ip_address="localhost")
    EMPTY = Empty()
    
    # 1. Setup custom policy
    joint_pos = robot.get_joint_positions()
    
    # Define your gains
    Kq = torch.ones(7) * 5.0
    Kqd = torch.ones(7) * 1.0
    Kx = torch.ones(6) * 50.0
    Kxd = torch.ones(6) * 5.0

    custom_policy = HybridJointImpedanceControl(
        joint_pos_current=joint_pos,
        Kq=Kq,
        Kqd=Kqd,
        Kx=Kx,
        Kxd=Kxd,
        robot_model=robot.robot_model 
    )

    print("Sending custom policy...")
    robot.send_torch_policy(custom_policy, blocking=False)

    # 2. Open gRPC stream
    st_stream = robot.grpc_connection.GetRobotStateStream(EMPTY)

    # 3. Setup CSV logging
    filename = f"wrench_log_{int(time.time())}.csv"
    print(f"Logging to {filename}... Press Ctrl+C to stop.")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
        
        try:
            for state_msg in st_stream:
                # 1. Extract current state from gRPC message
                q = torch.tensor(state_msg.joint_positions)
                dq = torch.tensor(state_msg.joint_velocities)
                
                # 2. Compute kinematics locally (Mirroring the controller)
                ee_pos, ee_quat = robot.robot_model.forward_kinematics(q)
                jacobian = robot.robot_model.compute_jacobian(q)
                ee_twist = jacobian @ dq
                
                # 3. Replicate the wrench calculation
                # We access the desired values directly from the policy parameters
                wrench_feedback = custom_policy.pose_pd(
                    ee_pos, 
                    ee_quat, 
                    ee_twist,
                    custom_policy.ee_pos_desired,
                    custom_policy.ee_quat_desired,
                    torch.cat([custom_policy.ee_vel_desired, custom_policy.ee_rvel_desired])
                )
                
                # 4. Log to CSV
                writer.writerow(wrench_feedback.tolist())
                
                # Optional: periodic print to verify values
                # if int(time.time() * 10) % 5 == 0: 
                print(f"Force: {wrench_feedback[:3].tolist()}")

        except KeyboardInterrupt:
            print("\nInterrupt caught. Terminating policy...")
            robot.terminate_current_policy()