import torch
from model import GSPAnomalyClassifier
from data import load_data, plot_features

# Load data and prepare tensors
X_tensor, y_tensor = load_data()

# Model and training setup
model = GSPAnomalyClassifier(input_dim=X_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCELoss()

# Training loop
for epoch in range(300):
    model.train()
    preds = model(X_tensor)
    loss = criterion(preds, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        with torch.no_grad():
            pred_class = (preds > 0.5).float()
            acc = (pred_class == y_tensor).float().mean()
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {acc.item()*100:.2f}%")

# Plot comparison
plot_features()

