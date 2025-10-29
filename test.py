"""
pDMP example adapted for EMG-driven phase (no autonomous replay).
- Phase only advances with user EMG activation.
- Direction flips with sign of intent (flex vs extend).
- Output (theta_d, dtheta_d) is ready for OIAC tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from ProjectInRobotics.pDMP.pDMP_functions import pDMP  # your class

# ----------------------- Experiment params -----------------------
dt        = 0.01         # 100 Hz typical
T         = 40.0
samples   = int(T/dt)

DOF       = 1            # 1-DoF elbow
N         = 25
alpha     = 25.0
beta      = alpha/4.0
lambd     = 0.995

# Elbow ROM (radians) and mid position
theta_min, theta_max = np.deg2rad([10, 140])
theta_mid  = 0.5*(theta_min + theta_max)
amp        = 0.5*(theta_max - theta_min)

# Phase parameters
phi        = np.array([0.0])      # start phase
kappa_phi  = 2*np.pi*0.6          # max ~0.6 Hz at full activation
eps        = 0.06                 # EMG deadband (~6% MVC)

# Keep a nominal tau for the transformation system; we won't clock phase with it
tau_nom    = 1.2

# ----------------------- Build pDMP -----------------------
dmp = pDMP(DOF=DOF, N=N, alpha=alpha, beta=beta, lambd=lambd, dt=dt)
dmp.set_period(np.array([tau_nom]))
dmp.g[:] = np.array([theta_mid])   # rhythmic DMP "center"
dmp.r[:] = np.array([amp])         # amplitude scale

# Optional: seed state near center
dmp.y[:] = np.array([theta_mid])
dmp.z[:] = np.array([0.0])

# ----------------------- EMG stubs -----------------------
# Replace these with your real filtered/MVC-normalized EMG in [0,1]
def get_emg_sample(t):
    """
    Demo: user does bursts of flex then extend.
    Return Ea (biceps), Eb (triceps) in [0,1].
    """
    # Simple schedule: 4 s flex, 2 s relax, 4 s extend, 2 s relax, repeat
    cycle = 12.0
    tt = t % cycle
    if 0.0 <= tt < 4.0:
        Ea, Eb = 0.35, 0.02     # flex
    elif 6.0 <= tt < 10.0:
        Ea, Eb = 0.02, 0.35     # extend
    else:
        Ea, Eb = 0.02, 0.02     # relaxed
    return Ea, Eb

# ----------------------- Data buffers (for plotting) -----------------------
ts, y_ref_log, y_dmp_log, phi_log, u_log = [], [], [], [], []

# ----------------------- MAIN LOOP -----------------------
teach = False       # True if you want to learn from a reference y(t)
update_online = False  # True if you provide a teaching signal U (e.g., from EMG-cost)

# Reference signal for LEARN mode (only for illustration)
y_old, dy_old = np.array([theta_mid]), np.array([0.0])

for k in range(samples):
    t = k*dt

    # ---- (1) EMG -> net intent and activation for phase ----
    Ea, Eb = get_emg_sample(t)   # <-- plug your filtered/MVC EMG here
    u = Ea - Eb                  # net intent (flex positive)
    a_phi = max(abs(u) - eps, 0.0)

    # ---- (2) Update phase only when active; freeze otherwise ----
    dphi = kappa_phi * a_phi * np.sign(u)
    phi[0] = (phi[0] + dphi * dt) % (2*np.pi)   # modulo 2π keeps oscillator bounded
    dmp.set_phase(phi)
    dmp.set_period(np.array([tau_nom]))         # period stays constant in transform

    # ---- (3) Choose DMP mode ----
    if teach:
        # Example: teach the DMP a sinusoid around theta_mid (for demonstration)
        y_ref = np.array([theta_mid + amp * np.sin(phi[0])])
        dy_ref = (y_ref - y_old) / dt
        ddy_ref = (dy_ref - dy_old) / dt
        dmp.learn(y_ref, dy_ref, ddy_ref)
        y_old, dy_old = y_ref, dy_ref
    elif update_online:
        # Example: unsupervised update by some teaching signal U (one DOF)
        U = np.array([0.0])      # replace with e.g. gradient signal
        dmp.update(U)
    else:
        dmp.repeat()             # evaluate with current weights

    # ---- (4) Integrate DMP to get desired trajectory ----
    dmp.integration()
    y_dmp, dy_dmp, _, _ = dmp.get_state()

    # Clamp to ROM (safety)
    y_dmp = np.clip(y_dmp, theta_min, theta_max)

    # ---- (5) Send to OIAC (hook) ----
    # oiac.track(theta_ref=y_dmp, dtheta_ref=dy_dmp)

    # ---- (6) Log for plotting ----
    ts.append(t)
    phi_log.append(phi[0])
    u_log.append(u)
    # For comparison, show a “reference” only if teach==True; else plot midline
    y_ref_log.append(theta_mid if not teach else y_ref[0])
    y_dmp_log.append(y_dmp[0])

# ----------------------- PLOTS -----------------------
ts = np.array(ts)
y_ref_log = np.array(y_ref_log)
y_dmp_log = np.array(y_dmp_log)
phi_log = np.array(phi_log)
u_log = np.array(u_log)

plt.figure()
plt.plot(ts, y_dmp_log, label='pDMP output (θ_d)')
plt.plot(ts, y_ref_log, '--', label='reference (for LEARN demo)' if teach else 'center')
plt.xlabel('time [s]'); plt.ylabel('angle [rad]'); plt.legend(); plt.title('EMG-driven pDMP (freezes without input)')

plt.figure()
plt.plot(ts, phi_log, label='phase φ')
plt.plot(ts, u_log, label='net intent u (Ea-Eb)')
plt.xlabel('time [s]'); plt.legend(); plt.title('Phase only moves with EMG')
plt.show()
