import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):
    """Same architecture as the notebook (30,840 params)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(500, 50)
        self.out = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 500)
        return self.out(F.relu(self.fc1(x)))
