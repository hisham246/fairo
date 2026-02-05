import torch
import csv
from polymetis import RobotInterface

if __name__ == "__main__":
    robot = RobotInterface(ip_address="localhost")
    print("Performing Cartesian impedance control...")
    
    # 1. Start the policy (this returns a generator)
    # Note: blocking=False allows us to enter the loop below immediately
    results = robot.start_cartesian_impedance()

    # 2. Open your CSV file here in the main script
    with open("wrench_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

        try:
            # 3. Iterate through the results generator
            # This loop runs as long as the policy is active
            for state_dict, return_dict in results:
                # Extract the log data we defined in the forward() method
                if "wrench_log" in return_dict:
                    wrench = return_dict["wrench_log"].tolist()
                    writer.writerow(wrench)
                
                # You can still perform updates here
                # robot.update_desired_ee_pose(...)
                
        except KeyboardInterrupt:
            print("Stopping...")
            robot.terminate_current_policy()