from torch.nn import Linear, Module
from torch import sigmoid


class AgeHead(Module):
    def __init__(self):
        super(AgeHead, self).__init__()
        self.fc1 = Linear(512, 100)

    def forward(self, x):
        x = self.fc1(x)
        x = sigmoid(x)

        return x
