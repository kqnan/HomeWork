import torch
import torch.nn as nn
from torchviz import make_dot

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

model = MyModel()
x = torch.randn(1, 10)
y = model(x)
make_dot(y, params=dict(model.named_parameters())).render("model", format="pdf")
