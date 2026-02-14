import numpy as np
import pandas as pd

def damped_pinv(A, lam=1e-4):
    # A: (m,n). returns (n,m)
    m, n = A.shape
    if m <= n:
        return A.T @ np.linalg.inv(A @ A.T + (lam**2) * np.eye(m))
    else:
        return np.linalg.inv(A.T @ A + (lam**2) * np.eye(n)) @ A.T

# --- load ---
df = pd.read_csv("/mnt/data/robot_state_1_episode_1.csv")

q_cols  = [f"joint_positions_{i}" for i in range(7)]
tau_cols = [f"motor_torques_external_{i}" for i in range(7)]

Q   = df[q_cols].to_numpy()
Tau = df[tau_cols].to_numpy()

# --- TODO: provide/instantiate your robot model here ---
# Example pattern (you may need to adjust import/constructor based on your stack):
from torchcontrol.models import FrankaModel
# robot_model = FrankaModel()

def jacobian_numpy(robot_model, q_np):
    # robot_model.compute_jacobian expects torch tensor in many implementations
    import torch
    q = torch.tensor(q_np, dtype=torch.float32)
    J = robot_model.compute_jacobian(q).detach().cpu().numpy()  # (6,7)
    return J

# --- compute wrench ---
wrench = np.zeros((len(df), 6))
lam = 1e-3  # increase if noisy / near singularities

for k in range(len(df)):
    J = jacobian_numpy(robot_model, Q[k])     # (6,7)
    JT = J.T                                  # (7,6)
    JT_pinv = damped_pinv(JT, lam=lam)        # (6,7)
    wrench[k] = JT_pinv @ Tau[k]              # (6,)

out = pd.DataFrame(wrench, columns=["Fx","Fy","Fz","Tx","Ty","Tz"])
out.insert(0, "timestamp", df["timestamp"].to_numpy())
out.to_csv("ee_wrench_estimate.csv", index=False)
print("Wrote ee_wrench_estimate.csv")
