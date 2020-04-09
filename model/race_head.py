from torch.nn import Linear, Module
from torch.nn.functional import log_softmax


class RaceHead(Module):
    def __init__(self, classnum=5):
        super(RaceHead, self).__init__()
        self.fc1 = Linear(512, classnum)

    def forward(self, x):
        x = self.fc1(x)
        x = log_softmax(x, dim=1)

        return x
