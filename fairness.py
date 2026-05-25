import numpy as np


def jains_fairness_index(values, eps: float = 1e-8) -> float:
    """Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 1.0
    x = np.maximum(x, 0.0)
    denominator = x.size * np.sum(np.square(x)) + eps
    return float((np.sum(x) ** 2) / denominator)


def delay_fairness(delays, eps: float = 1e-8) -> float:
    """Fairness over inverse delay, so smaller and more equal delays score higher."""
    d = np.asarray(delays, dtype=np.float64)
    service = 1.0 / (1.0 + np.maximum(d, 0.0))
    return jains_fairness_index(service, eps=eps)


def throughput_fairness(throughputs, eps: float = 1e-8) -> float:
    return jains_fairness_index(throughputs, eps=eps)

