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

    with open("wrench_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

        print("Logging... Press Ctrl+C to stop.")
        try:
            for state_msg in st_stream:
                # 3. Manually deserialize the 'extra_data'
                # Polymetis stores 'extra_data' as a Dict[str, Tensor] serialized via torch.jit
                if state_msg.extra_data:
                    # Use torch.jit.load on a buffer created from the raw bytes
                    buffer = io.BytesIO(state_msg.extra_data)
                    
                    # In many Polymetis versions, extra_data is a serialized ParamDictContainer
                    try:
                        # We load the data. Since it's a scripted dict, we can access it:
                        extra_data_container = torch.jit.load(buffer)
                        # Access the dictionary (usually named 'param_dict' or returned via forward)
                        return_dict = extra_data_container.forward()
                        
                        if "wrench_log" in return_dict:
                            wrench = return_dict["wrench_log"].tolist()
                            writer.writerow(wrench)
                    except Exception as e:
                        # If jit.load fails, the format might be raw bytes of the tensor
                        pass
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            robot.terminate_current_policy()