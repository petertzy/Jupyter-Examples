import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from utils import get_dataloader
from models.model import SimpleNet

# 生成数据
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# DataLoader
loader = get_dataloader(X, y)

# 模型
model = SimpleNet()

# 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []

# 训练
for epoch in range(20):
    total_loss = 0
    for xb, yb in loader:
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# 可视化 Loss
plt.plot(losses)
plt.title('Training Loss')
plt.show()
