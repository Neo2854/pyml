from torch import nn

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.network = None

    def forward(self, x):
        return self.network(x)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.network = None

    def forward(self, x):
        return self.network(x)