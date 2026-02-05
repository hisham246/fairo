import torch
import csv
from polymetis import RobotInterface
# Import your custom class from the file where you defined it
from torchcontrol.policies.impedance import HybridJointImpedanceControl 

if __name__ == "__main__":
    robot = RobotInterface(ip_address="localhost")
    
    # 1. Setup your custom policy
    joint_pos = robot.get_joint_positions()
    robot_model = robot.get_robot_model()
    
    # Initialize your specific class
    # Make sure to pass the gains/parameters your __init__ expects
    custom_policy = HybridJointImpedanceControl(
        joint_pos_current=joint_pos,
        Kq=torch.ones(7) * 50.0,
        Kqd=torch.ones(7) * 5.0,
        Kx=torch.ones(6) * 100.0,
        Kxd=torch.ones(6) * 10.0,
        robot_model=robot_model
    )

    print("Sending custom policy...")
    
    # 2. Use send_torch_policy to get the results generator
    # This is the key change!
    results = robot.send_torch_policy(custom_policy, blocking=False)

    if results is None:
        print("Error: Policy failed to start.")
        exit()

    # 3. Now the loop will work
    with open("wrench_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

        try:
            for state_dict, return_dict in results:
                # 'return_dict' contains the values you put in 'forward'
                if "wrench_log" in return_dict:
                    wrench = return_dict["wrench_log"].tolist()
                    writer.writerow(wrench)
                
        except KeyboardInterrupt:
            robot.terminate_current_policy()