import torch
from polymetis import RobotInterface

def test_polymetis_kinematics():
    print("Connecting to Polymetis server...")
    # Initialize the robot interface (make sure your server is running!)
    robot = RobotInterface(ip_address="localhost")
    
    print("\n--- 1. Forward Kinematics (FK) Test ---")
    # Get current joint positions
    current_joints = robot.get_joint_positions()
    
    # Ask Polymetis where it thinks the TCP is right now
    # This uses the ee_link_name from your config
    ee_pos, ee_quat = robot.robot_model.forward_kinematics(current_joints)
    
    print(f"Current Joint Angles: {current_joints}")
    print(f"Calculated TCP Position (X, Y, Z): {ee_pos}")
    
    print("\n--- 2. Inverse Kinematics (IK) Test ---")
    # Let's create a target exactly 5 cm (0.05m) straight UP (+Z) from the current TCP
    target_pos = ee_pos.clone()
    target_pos[0] += 0.2
    target_quat = ee_quat.clone() # Keep the exact same orientation
    
    print(f"Target TCP Position (5cm up): {target_pos}")
    
    # Ask the IK solver for the joint angles to reach this new TCP
    ik_joints = robot.robot_model.inverse_kinematics(target_pos, target_quat, rest_pose=current_joints)
    print(f"IK Calculated Joint Angles: {ik_joints}")
    
    print("\n--- 3. The Verification (Round Trip) ---")
    # Feed the IK joint angles BACK into the FK solver. 
    # If the IK worked perfectly for the TCP frame, this output should exactly match your target_pos!
    verify_pos, verify_quat = robot.robot_model.forward_kinematics(ik_joints)
    
    print(f"Verification Position: {verify_pos}")
    
    # Calculate the error (Euclidean distance)
    error = torch.norm(target_pos - verify_pos).item()
    print(f"IK Error: {error * 1000:.2f} millimeters")
    
    if error < 0.001:
        print("\nSUCCESS: Polymetis IK and FK are perfectly tracking your panda_hand_tcp frame!")
    else:
        print("\nWARNING: IK failed to converge, or the frame indices are mismatched in the config.")

if __name__ == "__main__":
    test_polymetis_kinematics()