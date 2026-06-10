import numpy as np
import matplotlib.pyplot as plt

# --- Setup ---
Z = 30.0          # true depth (unknown to controller)
f = 800.0         # focal length in pixels
dt = 0.01         # time step

# True Jacobian (what controller is trying to learn)
J_true = f / Z    # = 800/30 = 26.67 pixels per m/s

# Target pixel
p_star = 500.0

# Initial drone position
x_drone = -2.0    # meters sideways from camera axis
p = f * x_drone / Z + 320  # initial pixel (320 = image center)

# Controller estimates
J_bar = 10.0      # initial wrong estimate of Jacobian
alpha = 0.5       # control gain
delta = 0.1       # learning rate

# Memory for composite learning
memory_u = []
memory_dp = []

# Logs
pixel_log = []
J_bar_log = []
error_log = []

# --- Simulation loop ---
for i in range(1000):
    
    # Pixel error
    e = p_star - p
    
    # Control law: command = alpha * J_bar^-1 * error
    u = alpha * (1.0 / J_bar) * e
    
    # True pixel velocity (what camera actually sees)
    dp_true = J_true * u
    
    # --- Gradient descent update ---
    prediction_error = J_bar * u - dp_true
    J_bar = J_bar - delta * prediction_error * u
    
    # --- Drone moves ---
    x_drone += u * dt
    p = f * x_drone / Z + 320
    
    # Log
    pixel_log.append(p)
    J_bar_log.append(J_bar)
    error_log.append(e)

# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

ax1.plot(pixel_log)
ax1.axhline(p_star, color='r', linestyle='--', label='target')
ax1.set_ylabel('Pixel position')
ax1.legend()

ax2.plot(error_log)
ax2.axhline(0, color='r', linestyle='--')
ax2.set_ylabel('Pixel error')

ax3.plot(J_bar_log)
ax3.axhline(J_true, color='r', linestyle='--', label=f'true J = {J_true:.2f}')
ax3.set_ylabel('J̄ estimate')
ax3.legend()

plt.tight_layout()
plt.show()