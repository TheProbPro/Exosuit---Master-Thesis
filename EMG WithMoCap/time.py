import socket
import time
import numpy as np

HOST = ''
PORT = 5005

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((HOST, PORT))

offsets = []

NUM_SAMPLES = 50

print("Waiting for sync messages...")

for i in range(NUM_SAMPLES):
    data, addr = s.recvfrom(1024)

    t_receive = time.time()
    t_send = float(data.decode())

    offset = t_receive - t_send
    offsets.append(offset)

    print(f"[{i}] offset: {offset:.6f}")

# After collecting all samples
offsets = np.array(offsets)

print("\n--- RESULTS ---")
print("Min offset:", np.min(offsets))
print("Mean offset:", np.mean(offsets))
print("Max offset:", np.max(offsets))