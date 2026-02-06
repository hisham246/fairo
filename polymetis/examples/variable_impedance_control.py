import torch
import csv
import time
import matplotlib.pyplot as plt
from collections import deque
from polymetis import RobotInterface
from torchcontrol.policies.impedance import HybridJointImpedanceControl 
from polymetis_pb2 import Empty

if __name__ == "__main__":
    robot = RobotInterface(ip_address="localhost")
    EMPTY = Empty()
    
    # 1. Setup custom policy
    joint_pos = robot.get_joint_positions()
    custom_policy = HybridJointImpedanceControl(
        joint_pos_current=joint_pos,
        Kq=torch.ones(7) * 5.0,
        Kqd=torch.ones(7) * 1.0,
        Kx=torch.ones(6) * 50.0,
        Kxd=torch.ones(6) * 5.0,
        robot_model=robot.robot_model 
    )

    print("Sending custom policy...")
    robot.send_torch_policy(custom_policy, blocking=False)
    st_stream = robot.grpc_connection.GetRobotStateStream(EMPTY)

    # 2. Setup Plotting Data
    history_length = 200  # Number of points to show on x-axis
    force_data = [deque([0.0] * history_length, maxlen=history_length) for _ in range(3)]
    
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    lines = [ax.plot(list(force_data[i]), label=f'F{["x", "y", "z"][i]}')[0] for i in range(3)]
    ax.set_ylim(-20, 20)  # Adjust based on expected Newtons
    ax.set_ylabel("Force (N)")
    ax.legend(loc='upper right')
    plt.show()

    # 3. Logging & Plotting Loop
    filename = f"wrench_log_{int(time.time())}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
        
        count = 0
        try:
            for state_msg in st_stream:
                # Calculate Wrench (Mirroring controller)
                q = torch.tensor(state_msg.joint_positions)
                dq = torch.tensor(state_msg.joint_velocities)
                ee_pos, ee_quat = robot.robot_model.forward_kinematics(q)
                jacobian = robot.robot_model.compute_jacobian(q)
                ee_twist = jacobian @ dq
                
                wrench_feedback = custom_policy.pose_pd(
                    ee_pos, ee_quat, ee_twist,
                    custom_policy.ee_pos_desired,
                    custom_policy.ee_quat_desired,
                    torch.cat([custom_policy.ee_vel_desired, custom_policy.ee_rvel_desired])
                )
                
                # Save to file
                wrench_list = wrench_feedback.tolist()
                writer.writerow(wrench_list)

                # Update plot data every 20 iterations (~50Hz refresh rate)
                count += 1
                if count % 20 == 0:
                    for i in range(3):
                        force_data[i].append(wrench_list[i])
                        lines[i].set_ydata(list(force_data[i]))
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        except KeyboardInterrupt:
            print("\nShutting down...")
            robot.terminate_current_policy()
            plt.close()