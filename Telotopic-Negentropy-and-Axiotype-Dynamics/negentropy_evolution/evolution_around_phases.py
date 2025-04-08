import matplotlib.pyplot as plt

# Telotopic negentropy values across the four phases (based on extracted content)
phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
actor_a_negentropy = [0.864, 0.832, 0.854, 0.915]  # approximated from narrative
actor_b_negentropy = [0.148, 0.362, 0.598, 0.723]  # inferred from descriptions

# Plotting the figure
plt.figure(figsize=(10, 6))
plt.plot(phases, actor_a_negentropy, marker='o', label="Actor A", linewidth=2)
plt.plot(phases, actor_b_negentropy, marker='s', label="Actor B", linewidth=2)
plt.title("Telotopic Negentropy Trajectories Across Phases", fontsize=14)
plt.xlabel("Interaction Phase")
plt.ylabel("Telotopic Negentropy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
