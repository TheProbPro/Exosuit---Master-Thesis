import math
import matplotlib.pyplot as plt
import numpy as np

# --- Functions ---
def sine_position(step, speed=0.05, min_val=0, max_val=140):
    amplitude = (max_val - min_val) / 2
    offset = min_val + amplitude
    x = step * speed
    return amplitude * math.sin(x) + offset


def sine_velocity(step, speed=0.05, min_val=0, max_val=140):
    amplitude = (max_val - min_val) / 2
    x = step * speed
    return amplitude * speed * math.cos(x)


# --- Simulation parameters ---
duration_sec = 10.0
fps = 100                    # steps per second
num_steps = int(duration_sec * fps)

speed = 0.05
min_val = 0
max_val = 140

# --- Generate data ---
steps = np.arange(num_steps)
time = steps / fps

positions = [sine_position(s, speed, min_val, max_val) for s in steps]
velocities = [sine_velocity(s, speed, min_val, max_val) for s in steps]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

ax1.plot(time, positions)
ax1.set_title("Sine Position")
ax1.set_ylabel("Position")
ax1.grid(True)

ax2.plot(time, velocities)
ax2.set_title("Sine Velocity")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Velocity")
ax2.grid(True)

plt.tight_layout()
plt.show()
