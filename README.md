# FA-HMAPPO: Fairness-Aware Hierarchical Multi-Agent PPO

Research project:

**Queue-Aware Resource Management and Service Fairness in Multi-Tier Air-Ground Networks: A Hierarchical Multi-Agent Reinforcement Learning Approach**

This repository implements a complete executable simulator and MAPPO training pipeline for a UAV-assisted MEC network with IoT devices, UAV edge servers, and one HAPS upper-tier server.

## Folder Structure

```text
project/
├── agents.py          # Shared decentralized actor and centralized critic networks
├── config.py          # Environment, reward, training, and path configuration
├── environment.py     # Gymnasium air-ground MEC simulation environment
├── evaluate.py        # Checkpoint loading and test-time evaluation
├── fairness.py        # Jain fairness index utilities
├── mappo.py           # Rollout buffer, GAE, PPO clipping, save/load
├── plots.py           # Training metric visualization
├── requirements.txt   # Python dependencies
├── reward.py          # Weighted FA-HMAPPO reward function
├── train.py           # End-to-end training entry point
└── utils.py           # Seeding, paths, JSON, device helpers
```

Generated outputs:

```text
project/checkpoints/       # PPO checkpoints
project/results/           # CSV metrics and run config
project/figures/           # Reward, delay, energy, fairness, throughput plots
```

## System Model

The network has `N` IoT devices, `M` UAV MEC servers, and one HAPS server:

```text
IoT Devices -> UAV MEC Layer -> HAPS Computing Layer
```

At each time slot, IoT device `i` receives a stochastic task arrival with input size `b_i(t)` bits and CPU intensity `c_i(t)` cycles/bit. Each device keeps a queue `Q_i(t)`. UAVs move toward queue hotspots, creating a dynamic air-ground topology.

Queue evolution:

```text
Q_i(t+1) = min(Q_max, max(Q_i(t) - S_i(t), 0) + A_i(t))
```

where `A_i(t)` is the arrived task size and `S_i(t)` is the number of processed bits in the slot.

Wireless uplink rate from device `i` to UAV `m`:

```text
R_{i,m}(t) = B_i log2(1 + P_i g_{i,m}(t) / N0)
g_{i,m}(t) = d_{i,m}(t)^(-lambda)
```

where `B_i` is the shared bandwidth, `P_i` is transmit power, `N0` is noise power, and `lambda` is the path-loss exponent.

## State Space

Each IoT device is one learning agent. Its local observation includes:

1. Queue length
2. Last task size
3. Local CPU ratio
4. UAV CPU ratio
5. Device x-position
6. Device y-position
7. Nearest UAV x-position
8. Nearest UAV y-position
9. Distance to nearest UAV
10. Bandwidth availability
11. Remaining energy
12. Previous delay
13. Previous throughput
14. Nearest UAV queue load
15. Normalized time index

The centralized critic receives the concatenated observations of all agents.

## Action Space

Each agent outputs a continuous action vector:

```text
a_i(t) = [rho_i, s_i, r_i, p_i]
```

where:

- `rho_i`: offloading ratio
- `s_i`: target selector, mapped to one UAV or HAPS
- `r_i`: CPU resource request weight
- `p_i`: scheduling priority

The environment decodes these continuous PPO actions into offloading, UAV/HAPS selection, CPU allocation, and scheduling behavior.

## Reward Function

FA-HMAPPO uses a fairness-aware weighted objective:

```text
R(t) =
- alpha * D_queue(t)
- beta  * E(t)
+ gamma * J(t)
+ delta * U(t)
+ eta   * T(t)
- zeta  * H(t)
```

where:

- `D_queue(t)` is average normalized queue delay
- `E(t)` is normalized energy consumption
- `J(t)` is fairness
- `U(t)` is resource utilization
- `T(t)` is throughput
- `H(t)` is hotspot imbalance across UAVs

Jain's fairness index is:

```text
J(x) = (sum_i x_i)^2 / (N * sum_i x_i^2)
```

The implementation combines delay fairness over inverse delay and throughput fairness.

## MARL Algorithm

The implementation follows MAPPO with:

- Parameter-shared decentralized actor for IoT agents
- Centralized critic over the global state
- Generalized Advantage Estimation
- PPO clipped policy objective
- Entropy regularization
- Checkpoint save/load support

Policy objective:

```text
L_clip(theta) =
E[min(r_t(theta) A_t, clip(r_t(theta), 1-epsilon, 1+epsilon) A_t)]
```

with:

```text
r_t(theta) = pi_theta(a_t | o_t) / pi_theta_old(a_t | o_t)
```

## Installation

From the `project` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Run a quick smoke training:

```bash
python train.py --episodes 5 --devices 8 --uavs 2 --log-every 1
```

Run a fuller experiment:

```bash
python train.py --episodes 250 --devices 12 --uavs 3 --log-every 10
```

The final checkpoint is saved to:

```text
checkpoints/fahmappo_final.pt
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/fahmappo_final.pt --episodes 20
```

If you trained with a custom topology, pass the same size:

```bash
python evaluate.py --checkpoint checkpoints/fahmappo_final.pt --episodes 20 --devices 8 --uavs 2
```

Evaluation writes:

```text
results/evaluation_metrics.csv
```

## Plots

Training automatically generates:

- `figures/reward.png`
- `figures/queue_delay.png`
- `figures/energy.png`
- `figures/fairness.png`
- `figures/throughput.png`
- `figures/resource_utilization.png`

You can regenerate them with:

```bash
python plots.py
```

## Expected Outputs

After training, expect:

1. Episode logs with reward, delay, fairness, and throughput
2. PPO checkpoints under `checkpoints/`
3. Training metrics under `results/training_metrics.csv`
4. Evaluation metrics under `results/evaluation_metrics.csv`
5. Publication-style learning curves under `figures/`

This code is designed as a research-grade baseline: the simulator is transparent, reward terms are modular, and the algorithm can be extended with discrete action heads, explicit UAV-control agents, baselines such as random/greedy offloading, or PettingZoo wrappers.