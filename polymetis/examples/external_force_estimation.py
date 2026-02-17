#!/usr/bin/env python3
import time
import grpc

import polymetis_pb2 as pb2
import polymetis_pb2_grpc as pb2_grpc


def main():
    server_address = "127.0.0.1:50051"  # controller server
    channel = grpc.insecure_channel(server_address)
    stub = pb2_grpc.PolymetisControllerServerStub(channel)

    while True:
        state = stub.GetRobotState(pb2.Empty())

        wrench = list(state.external_wrench_base)  # [Fx,Fy,Fz,Tx,Ty,Tz]
        tau_ext = list(state.motor_torques_external)  # tau_ext_hat_filtered (7)

        # Wrench should be length 6 if everything is wired correctly
        print(
            f"wrench_size={len(wrench)} wrench={wrench} | "
            f"tau_ext_size={len(tau_ext)} tau_ext(first3)={tau_ext[:3]}"
        )

        time.sleep(0.1)  # 10 Hz


if __name__ == "__main__":
    main()