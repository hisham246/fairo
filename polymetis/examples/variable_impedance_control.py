import torch
import csv
import io
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

    # 2. Open stream
    st_stream = robot.grpc_connection.GetRobotStateStream(EMPTY)

    print("--- Message Inspection ---")
    try:
        for state_msg in st_stream:
            # List all fields available in the message to find the right one
            fields = [field[0].name for field in state_msg.ListFields()]
            print(f"Available fields in RobotState: {fields}")
            
            # Check for common alternative names for the return dict
            target_field = None
            for candidate in ["custom_data", "extra_data", "policy_data"]:
                if hasattr(state_msg, candidate):
                    target_field = candidate
                    break
            
            if target_field:
                print(f"Found data in: {target_field}")
                # Once found, we can try to decode it
                data_bytes = getattr(state_msg, target_field)
                if data_bytes:
                    buffer = io.BytesIO(data_bytes)
                    # Attempt to load
                    try:
                        data = torch.jit.load(buffer)
                        print("Successfully decoded data!")
                        # Break after one successful inspection to avoid flooding
                        break 
                    except:
                        print("Found field but failed to decode via torch.jit.load")
            else:
                print("Could not find a binary data field in this message version.")
                break

    except KeyboardInterrupt:
        robot.terminate_current_policy()