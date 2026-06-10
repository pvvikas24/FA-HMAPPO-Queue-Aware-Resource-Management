import os
import matplotlib.pyplot as plt

# ==========================
# RESULTS
# ==========================

methods = ["Greedy", "Random", "FA-HMAPPO"]

reward = [46.0663, 74.6908, 146.6406]
delay = [0.9327, 0.6196, 0.1618]
energy = [0.0513, 0.0473, 0.0374]
fairness = [0.9732, 0.7677, 0.8124]
throughput = [0.3340, 0.3512, 0.3623]
utilization = [0.2469, 0.2628, 0.2811]

# ==========================
# CREATE OUTPUT FOLDER
# ==========================

os.makedirs("comparison_graphs", exist_ok=True)

# ==========================
# FUNCTION
# ==========================

def create_graph(values, title, ylabel, filename):
    
    plt.figure(figsize=(10, 5))

    plt.plot(
        methods,
        values,
        marker='o',
        linewidth=3,
        markersize=10,
        label="Comparison"
    )

    for i, v in enumerate(values):
        plt.text(
            i,
            v,
            f"{v:.4f}",
            fontsize=10,
            ha='center',
            va='bottom'
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Methods", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"comparison_graphs/{filename}.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ==========================
# INDIVIDUAL COMPARISON GRAPHS
# ==========================

create_graph(
    reward,
    "Reward Comparison",
    "Reward",
    "reward_comparison"
)

create_graph(
    delay,
    "Queue Delay Comparison",
    "Queue Delay",
    "delay_comparison"
)

create_graph(
    energy,
    "Energy Consumption Comparison",
    "Energy",
    "energy_comparison"
)

create_graph(
    fairness,
    "Fairness Comparison",
    "Fairness Index",
    "fairness_comparison"
)

create_graph(
    throughput,
    "Throughput Comparison",
    "Throughput",
    "throughput_comparison"
)

create_graph(
    utilization,
    "Resource Utilization Comparison",
    "Resource Utilization",
    "utilization_comparison"
)

# ==========================
# COMBINED FIGURE
# ==========================

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

metrics = [
    ("Reward", reward),
    ("Queue Delay", delay),
    ("Energy", energy),
    ("Fairness", fairness),
    ("Throughput", throughput),
    ("Resource Utilization", utilization),
]

for ax, (title, values) in zip(axs.flat, metrics):

    ax.plot(
        methods,
        values,
        marker='o',
        linewidth=3,
        markersize=8
    )

    for i, v in enumerate(values):
        ax.text(
            i,
            v,
            f"{v:.3f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.suptitle(
    "Performance Comparison: Greedy vs Random vs FA-HMAPPO",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "comparison_graphs/all_comparison_metrics.png",
    dpi=300
)

plt.show()

plt.close()

print("\nGraphs saved successfully.")
print("Folder: comparison_graphs/")