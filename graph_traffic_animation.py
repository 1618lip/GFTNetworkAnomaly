import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
n_nodes = 20
anomaly_node = 5
anomaly_frequency = 0.2
n_timesteps = 200
seed = 42
np.random.seed(seed)

# Create graph and layout
G = nx.connected_watts_strogatz_graph(n=n_nodes, k=4, p=0.3, seed=seed)
pos = nx.spring_layout(G, seed=seed)

# Generate traffic data
def generate_traffic(inject=False):
    base = np.random.poisson(lam=4, size=(n_timesteps, n_nodes))
    if inject:
        t = np.arange(n_timesteps)
        pulse = 15 * np.sin(2 * np.pi * anomaly_frequency * t)
        pulse = np.maximum(pulse, 0)
        base[:, anomaly_node] += pulse.astype(int)
    return base

traffic_data = generate_traffic(inject=True)

# Setup figure
fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    traffic = traffic_data[frame]
    node_colors = traffic / np.max(traffic)
    nx.draw(G, pos, node_color=node_colors, cmap='inferno', with_labels=True, node_size=500, ax=ax, vmin=0, vmax=1)
    ax.set_title(f"Graph Traffic - Timestep {frame}", fontsize=14)

ani = animation.FuncAnimation(fig, update, frames=n_timesteps, interval=70)

# Save animation
ani.save("gsp_graph_traffic.mp4", writer='ffmpeg')

