import pybullet as p
import pybullet_data
import time

def get_link_index(robot_id, link_name):
    """Finds the PyBullet link index for a given link name."""
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[12].decode('utf-8') == link_name:
            return i
    return -1

def main():
    physicsClient = p.connect(p.GUI)
    # --- NEW: Disable collision mesh (wireframe) visualization ---
    # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    # -------------------------------------------------------------
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    
    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    print("Loading Panda URDF...")
    try:
        # Note: I left your absolute path in here
        robotId = p.loadURDF("/home/hisham246/uwaterloo/fairo/polymetis/polymetis/data/franka_panda/panda_arm.urdf", start_pos, start_orientation, useFixedBase=True)
        print("Successfully loaded the robot!")

        # --- NEW CODE: Hide the collision-only sub-links ---
        # Loop through all the joints/links in the robot
        for i in range(p.getNumJoints(robotId)):
            joint_info = p.getJointInfo(robotId, i)
            link_name = joint_info[12].decode('utf-8')
            
            # If the link is a self-collision link (ends with '_sc')
            if link_name.endswith('_sc'):
                # Change its color to be fully transparent (Alpha = 0)
                p.changeVisualShape(robotId, i, rgbaColor=[1, 1, 1, 0])
        
        # 1. Setup the Home Pose
        home_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
        revolute_joint_indices = []
        for i in range(p.getNumJoints(robotId)):
            if p.getJointInfo(robotId, i)[2] == p.JOINT_REVOLUTE:
                revolute_joint_indices.append(i)
        
        print("Applying motor controls to fight gravity...")
        for i, joint_idx in enumerate(revolute_joint_indices):
            if i < len(home_pose):
                p.resetJointState(robotId, joint_idx, home_pose[i])
                p.setJointMotorControl2(bodyIndex=robotId, jointIndex=joint_idx, 
                                        controlMode=p.POSITION_CONTROL, targetPosition=home_pose[i], force=250)
        
        # 2. Get the link index for panda_link8
        link8_index = get_link_index(robotId, "panda_link8")
        
        # 3. Create dummy lines at the origin first to get their Unique IDs
        # We will update these IDs inside the loop
        line_x = p.addUserDebugLine([0,0,0], [0,0,0], [1,0,0], lineWidth=3)
        line_y = p.addUserDebugLine([0,0,0], [0,0,0], [0,1,0], lineWidth=3)
        line_z = p.addUserDebugLine([0,0,0], [0,0,0], [0,0,1], lineWidth=3)
        axis_len = 0.15

    except p.error as e:
        print(f"\n[ERROR] Failed to load URDF: {e}\n")
        p.disconnect()
        return

    print("Simulation running. Press Ctrl+C in the terminal to exit.")
    try:
        while True:
            # --- NEW CODE: Dynamically update the reference frame ---
            if link8_index != -1:
                # getLinkState returns multiple things. 
                # Index 4 is the URDF Link Frame Position, Index 5 is the URDF Link Frame Orientation
                state = p.getLinkState(robotId, link8_index)
                pos = state[4]
                ori = state[5]

                # Use PyBullet's built-in math to transform the local axes to world coordinates
                x_tip, _ = p.multiplyTransforms(pos, ori, [axis_len, 0, 0], [0,0,0,1])
                y_tip, _ = p.multiplyTransforms(pos, ori, [0, axis_len, 0], [0,0,0,1])
                z_tip, _ = p.multiplyTransforms(pos, ori, [0, 0, axis_len], [0,0,0,1])

                # Replace the old lines with the new coordinates
                p.addUserDebugLine(pos, x_tip, [1,0,0], lineWidth=3, replaceItemUniqueId=line_x)
                p.addUserDebugLine(pos, y_tip, [0,1,0], lineWidth=3, replaceItemUniqueId=line_y)
                p.addUserDebugLine(pos, z_tip, [0,0,1], lineWidth=3, replaceItemUniqueId=line_z)
            # --------------------------------------------------------

            p.stepSimulation()
            time.sleep(1. / 240.) 
            
    except KeyboardInterrupt:
        print("\nExiting visualization...")
        
    p.disconnect()

if __name__ == '__main__':
    main()


# import pybullet as p

# def main():
#     p.connect(p.DIRECT) # Run without GUI just to get the index
#     robotId = p.loadURDF("/home/hisham246/uwaterloo/fairo/polymetis/polymetis/data/franka_panda/panda_arm.urdf", useFixedBase=True)
    
#     target_link = "panda_link8"
    
#     for i in range(p.getNumJoints(robotId)):
#         info = p.getJointInfo(robotId, i)
#         link_name = info[12].decode('utf-8')
#         print(i, link_name)  # Print the link name for debugging

#         if link_name == target_link:
#             print(f"\nSUCCESS! The index for {target_link} is: {i}\n")
#             p.disconnect()
#             return
            
#     print("Link not found!")
#     p.disconnect()

# if __name__ == '__main__':
#     main()