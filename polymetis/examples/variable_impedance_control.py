import torch
import csv
from polymetis import RobotInterface
from torchcontrol.policies.impedance import HybridJointImpedanceControl 
from polymetis_pb2 import Empty

if __name__ == "__main__":
    # Initialize robot
    robot = RobotInterface(ip_address="localhost")
    EMPTY = Empty()
    
    # 1. Setup custom policy
    joint_pos = robot.get_joint_positions()
    
    # Note: Use robot.robot_model (the attribute created in RobotInterface.__init__)
    custom_policy = HybridJointImpedanceControl(
        joint_pos_current=joint_pos,
        Kq=torch.ones(7) * 5.0,
        Kqd=torch.ones(7) * 1.0,
        Kx=torch.ones(6) * 50.0,
        Kxd=torch.ones(6) * 5.0,
        robot_model=robot.robot_model 
    )

    print("Sending custom policy...")
    
    # 2. Start the policy (non-blocking)
    # This sends it to the server, but as we saw in your code, this returns None
    robot.send_torch_policy(custom_policy, blocking=False)

    # 3. Connect to the server's feedback stream manually
    # This is where return_dict values (like 'wrench_log') are sent
    print("Opening stream and logging to wrench_log.csv...")
    st_stream = robot.grpc_connection.GetRobotStateStream(EMPTY)

    with open("wrench_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

        try:
            for state_dict_msg in st_stream:
                # The 'state_dict_msg' contains the 'extra_data' field 
                # where your return_dict values are stored as tensors
                
                # Convert the Protobuf map to a dictionary of torch tensors
                # Note: Polymetis handles the conversion from the msg format
                import torch
                from polymetis.utils.data import msg_to_tensors
                
                return_dict = msg_to_tensors(state_dict_msg.extra_data)

                if "wrench_log" in return_dict:
                    wrench = return_dict["wrench_log"].tolist()
                    writer.writerow(wrench)
                    # Optional: print to see it working
                    # print(f"Wrench: {wrench}")
                
        except KeyboardInterrupt:
            print("Interrupt caught. Terminating policy...")
            robot.terminate_current_policy()