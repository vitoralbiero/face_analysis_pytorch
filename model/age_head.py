from torch.nn import Linear, Module


class AgeHead(Module):
    def __init__(self):
        super(AgeHead, self).__init__()
        self.fc1 = Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)

        return x.squeeze()
