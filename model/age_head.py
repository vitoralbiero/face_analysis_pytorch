from torch import sigmoid
from torch.nn import Linear, Module


class AgeHead(Module):
    def __init__(self, classnum=100):
        super(AgeHead, self).__init__()
        self.fc1 = Linear(512, classnum)

    def forward(self, x):
        x = self.fc1(x)
        x = sigmoid(x)

        return x
