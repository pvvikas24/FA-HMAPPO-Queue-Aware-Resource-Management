import matplotlib.pyplot as plt
import numpy as np

methods = ["Greedy", "Random", "FA-HMAPPO"]

reward = [46.0663, 74.6908, 146.6406]
delay = [0.9327, 0.6196, 0.1618]
energy = [0.0513, 0.0473, 0.0374]
fairness = [0.9732, 0.7677, 0.8124]
throughput = [0.3340, 0.3512, 0.3623]
utilization = [0.2469, 0.2628, 0.2811]

fig, axs = plt.subplots(2, 3, figsize=(15, 8))

metrics = [
    ("Reward", reward),
    ("Queue Delay", delay),
    ("Energy", energy),
    ("Fairness", fairness),
    ("Throughput", throughput),
    ("Resource Utilization", utilization)
]

for ax, (title, values) in zip(axs.flat, metrics):
    bars = ax.bar(methods, values)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            h,
            f"{h:.3f}",
            ha='center',
            va='bottom'
        )

    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("Performance Comparison of Different Methods")
plt.tight_layout()
plt.savefig("comparison_results.png", dpi=300)
plt.show()