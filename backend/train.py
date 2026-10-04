"""Train the CNN.  Usage: python train.py path/to/mnist_train.csv  ->  mnist_cnn.pt"""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import MnistNet

# header=None + to_numeric/dropna works whether or not the CSV has a header row
df = pd.read_csv(sys.argv[1], header=None).apply(pd.to_numeric, errors="coerce").dropna()
data = df.to_numpy(dtype=np.float32)
print(f"loaded {data.shape[0]} images")
labels, data = data[:, 0], data[:, 1:]
data = (data / data.max()).reshape(-1, 1, 28, 28)

X = torch.tensor(data).float(); y = torch.tensor(labels).long()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.1)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True, drop_last=True)

def augment(x):
    """Random rotation (+-15 deg), scale (0.8-1.2), shift (~2px) per image."""
    n = x.size(0)
    ang = (torch.rand(n) - .5) * 2 * np.radians(15)
    sc = 1 + (torch.rand(n) - .5) * .4
    tx, ty = (torch.rand(n) - .5) * .3, (torch.rand(n) - .5) * .3
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([c/sc, -s/sc, tx], 1),
                         torch.stack([s/sc,  c/sc, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)

net = MnistNet(); lossfun = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=.001)

for epoch in range(15):
    net.train()
    for xb, yb in loader:
        loss = lossfun(net(augment(xb)), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = 100 * (net(Xte).argmax(1) == yte).float().mean().item()
    print(f"epoch {epoch+1}/15  loss {loss.item():.3f}  test acc {acc:.2f}%")

torch.save(net.state_dict(), "mnist_cnn.pt")
