import matplotlib.pyplot as plt
import numpy as np

# Redéfinir les angles et paramètres
theta_base_deg = 30
delta_theta_deg = 50
beta = 0.3
gamma = (1 - beta) / 2
theta_final = theta_base_deg * gamma + delta_theta_deg * (1 - abs(beta))

# Convertir en radians
theta_base_rad = np.deg2rad(theta_base_deg)
delta_theta_rad = np.deg2rad(delta_theta_deg)
theta_final_rad = np.deg2rad(theta_final)

# Origine
origin = [0], [0]

# Vecteurs
base_vector = [np.cos(theta_base_rad), np.sin(theta_base_rad)]
delta_vector = [np.cos(delta_theta_rad), np.sin(delta_theta_rad)]
final_vector = [np.cos(theta_final_rad), np.sin(theta_final_rad)]

# Tracer
plt.figure(figsize=(8, 8))
plt.quiver(*origin, *base_vector, color='blue', angles='xy', scale_units='xy', scale=1, label='θ_base (axiotype)')
plt.quiver(*origin, *delta_vector, color='green', angles='xy', scale_units='xy', scale=1, label='Δθ (contextual deviation)')
plt.quiver(*origin, *final_vector, color='red', angles='xy', scale_units='xy', scale=1, label='θ (resulting angle)', linewidth=2)

# Ajustements visuels
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.title('Construction of Affective Projection Angle θ')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
