import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        a1 = torch.sigmoid(self.layer1(x))
        a2 = torch.sigmoid(self.layer2(a1))
        return a2


X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])
model = TwoLayerNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(1000):
    a2 = model(X)
    loss = criterion(a2,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss.item())