#!/usr/bin/env python3
# Real-time plot of Polymetis Franka external wrench (external_wrench_base)
# - Figure 1: Fx, Fy, Fz (N)
# - Figure 2: Tx, Ty, Tz (Nm)
#
# Notes:
# - This polls GetRobotState at ~plot_hz (default 20 Hz).
# - Each chart is its own figure (no subplots).
# - Close the windows (or Ctrl+C) to stop.

import time
from collections import deque

import grpc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import polymetis_pb2 as pb2
import polymetis_pb2_grpc as pb2_grpc


def main():
    server_address = "127.0.0.1:50051"

    window_sec = 10.0     # how much history to show
    plot_hz = 20.0        # polling/plot update rate
    dt = 1.0 / plot_hz
    maxlen = int(window_sec * plot_hz)

    # Data buffers
    t0 = time.time()
    ts = deque(maxlen=maxlen)
    fx = deque(maxlen=maxlen)
    fy = deque(maxlen=maxlen)
    fz = deque(maxlen=maxlen)
    tx = deque(maxlen=maxlen)
    ty = deque(maxlen=maxlen)
    tz = deque(maxlen=maxlen)

    # gRPC client
    channel = grpc.insecure_channel(server_address)
    stub = pb2_grpc.PolymetisControllerServerStub(channel)

    # Figure 1: Forces
    figF, axF = plt.subplots()
    axF.set_title("External force (base frame)")
    axF.set_xlabel("t (s)")
    axF.set_ylabel("Force (N)")
    line_fx, = axF.plot([], [], label="Fx")
    line_fy, = axF.plot([], [], label="Fy")
    line_fz, = axF.plot([], [], label="Fz")
    axF.legend(loc="upper right")

    # Figure 2: Torques
    figT, axT = plt.subplots()
    axT.set_title("External torque (base frame)")
    axT.set_xlabel("t (s)")
    axT.set_ylabel("Torque (Nm)")
    line_tx, = axT.plot([], [], label="Tx")
    line_ty, = axT.plot([], [], label="Ty")
    line_tz, = axT.plot([], [], label="Tz")
    axT.legend(loc="upper right")

    last_ok = {"count": 0}

    def fetch_one():
        state = stub.GetRobotState(pb2.Empty())
        w = list(state.external_wrench_base)  # [Fx,Fy,Fz,Tx,Ty,Tz]
        return w

    def update(_frame_idx):
        # Pull one sample
        try:
            w = fetch_one()
        except Exception:
            # If the server hiccups, just skip this frame
            return ()

        if len(w) != 6:
            # Not populated yet; skip plotting until it is
            return ()

        now = time.time() - t0
        ts.append(now)
        fx.append(w[0]); fy.append(w[1]); fz.append(w[2])
        tx.append(w[3]); ty.append(w[4]); tz.append(w[5])
        last_ok["count"] += 1

        # Update lines
        x = list(ts)

        line_fx.set_data(x, list(fx))
        line_fy.set_data(x, list(fy))
        line_fz.set_data(x, list(fz))

        line_tx.set_data(x, list(tx))
        line_ty.set_data(x, list(ty))
        line_tz.set_data(x, list(tz))

        # Auto-scale axes nicely to current window
        axF.relim()
        axF.autoscale_view()

        axT.relim()
        axT.autoscale_view()

        return (line_fx, line_fy, line_fz, line_tx, line_ty, line_tz)

    # Animations (one per figure)
    animF = FuncAnimation(figF, update, interval=int(1000 * dt), blit=False)
    animT = FuncAnimation(figT, update, interval=int(1000 * dt), blit=False)

    plt.show()


if __name__ == "__main__":
    main()
