import numpy as np
import pandas as pd

# Parameters
n_nodes = 20
anomaly_node = 5
anomaly_frequency = 0.2
n_timesteps = 200
seed = 42
np.random.seed(seed)

# Simulate traffic data
def generate_traffic(inject=False):
    base = np.random.poisson(lam=4, size=(n_timesteps, n_nodes))
    if inject:
        t = np.arange(n_timesteps)
        pulse = 15 * np.sin(2 * np.pi * anomaly_frequency * t)
        pulse = np.maximum(pulse, 0)
        base[:, anomaly_node] += pulse.astype(int)
    return base

traffic_data = generate_traffic(inject=True)

# Create log
log_entries = []
for t in range(n_timesteps):
    entry = {"timestep": t}
    for n in range(n_nodes):
        entry[f"node_{n}"] = traffic_data[t, n]
    log_entries.append(entry)

df = pd.DataFrame(log_entries)
df.to_csv("graph_traffic_log.csv", index=False)
