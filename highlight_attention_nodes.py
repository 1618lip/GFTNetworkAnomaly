import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Parameters
n_nodes = 20
anomaly_frequency = 0.2
n_timesteps = 200
attention_threshold = 12
seed = 42
np.random.seed(seed)

# Create graph and layout
G = nx.connected_watts_strogatz_graph(n=n_nodes, k=4, p=0.3, seed=seed)
pos = nx.spring_layout(G, seed=seed)

# Generate traffic data
def generate_random_anomaly_traffic():
    base = np.random.poisson(lam=4, size=(n_timesteps, n_nodes))
    attacked_node = np.random.randint(0, n_nodes)  # RANDOM node
    print(f"Injected anomaly at node {attacked_node}")
    t = np.arange(n_timesteps)
    pulse = 15 * np.sin(2 * np.pi * anomaly_frequency * t)
    base[:, attacked_node] += np.maximum(pulse, 0).astype(int)
    return base, attacked_node

traffic_data, attacked_node = generate_random_anomaly_traffic()

# Generate log
log_entries = []
attention_nodes_per_frame = []

for t in range(n_timesteps):
    row = {"timestep": t}
    attention_nodes = []
    for node in range(n_nodes):
        val = traffic_data[t, node]
        row[f"node_{node}"] = val
        if val > attention_threshold:
            attention_nodes.append(node)
    attention_nodes_per_frame.append(attention_nodes)
    log_entries.append(row)

# Save CSV log
df = pd.DataFrame(log_entries)
df.to_csv("graph_traffic_log_random_anomaly.csv", index=False)

# Animation setup
fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    traffic = traffic_data[frame]
    attention = attention_nodes_per_frame[frame]
    node_colors = [
        "red" if i in attention else plt.cm.inferno(traffic[i] / np.max(traffic))
        for i in range(n_nodes)
    ]
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, ax=ax)
    ax.set_title(f"Timestep {frame} — Attention Nodes: {attention}", fontsize=11)

ani = animation.FuncAnimation(fig, update, frames=n_timesteps, interval=70)
ani.save("gsp_graph_traffic_random_anomaly.mp4", writer='ffmpeg')
