import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

# Graph construction
n_nodes = 20
n_timesteps = 200
anomaly_node = 5
anomaly_frequency = 0.2
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

G = nx.connected_watts_strogatz_graph(n=n_nodes, k=4, p=0.3, seed=seed)
L = nx.laplacian_matrix(G).todense()
L_torch = torch.tensor(L, dtype=torch.float32)
eigvals, eigvecs = torch.linalg.eigh(L_torch)

def gft(x, eigvecs):
    return eigvecs.T @ x

def generate_traffic(inject=False):
    base = np.random.poisson(lam=4, size=(n_timesteps, n_nodes))
    if inject:
        t = np.arange(n_timesteps)
        pulse = 15 * np.sin(2 * np.pi * anomaly_frequency * t)
        pulse = np.maximum(pulse, 0)
        base[:, anomaly_node] += pulse.astype(int)
    return base

def extract_features(data):
    features = []
    for snapshot in data:
        x = torch.tensor(snapshot, dtype=torch.float32)
        x_hat = gft(x, eigvecs)
        spec_energy = x_hat ** 2
        low_energy = torch.sum(spec_energy[:5])
        high_energy = torch.sum(spec_energy[-5:])
        ratio = high_energy / (low_energy + 1e-6)
        mean_energy = torch.mean(spec_energy)
        max_energy = torch.max(spec_energy)
        std_energy = torch.std(spec_energy)
        kurt = kurtosis(spec_energy.numpy())
        skw = skew(spec_energy.numpy())
        features.append([low_energy.item(), high_energy.item(), ratio.item(),
                         mean_energy.item(), max_energy.item(), std_energy.item(), kurt, skw])
    return features

def load_data():
    normal_data = generate_traffic(inject=False)
    anomaly_data = generate_traffic(inject=True)
    Xn = extract_features(normal_data)
    Xa = extract_features(anomaly_data)
    X = np.array(Xn + Xa)
    y = np.array([0]*n_timesteps + [1]*n_timesteps)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X_tensor, y_tensor

def plot_features():
    normal_data = generate_traffic(inject=False)
    anomaly_data = generate_traffic(inject=True)
    Xn = extract_features(normal_data)
    Xa = extract_features(anomaly_data)
    plt.figure(figsize=(10,5))
    plt.plot([f[2] for f in Xn], label='Normal')
    plt.plot([f[2] for f in Xa], label='Anomalous')
    plt.xlabel("Time")
    plt.ylabel("High/Low Energy Ratio")
    plt.title("Spectral Feature Comparison")
    plt.legend()
    plt.show()

