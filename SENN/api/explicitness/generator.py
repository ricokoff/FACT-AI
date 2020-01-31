import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, conceptizer):
        super(Generator, self).__init__()
        self.conceptizer = conceptizer
        self.generated = None

    def initialize(self, x=None):
        if x is None:
            self.generated = nn.Parameter(torch.normal(mean=torch.zeros(28, 28),
                                                       std=torch.ones(28, 28)), requires_grad=True)
        else:
            self.generated = nn.Parameter(x, requires_grad=True)

    def forward(self):
        activations = self.conceptizer.encode(self.generated.view(1, 1, 28, 28))
        return activations.squeeze()
