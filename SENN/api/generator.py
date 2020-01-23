import torch.nn as nn 
import torch

class Generator(nn.Module):
    def __init__(self, conceptizer):
        super(Generator, self).__init__()
    
        self.conceptizer = conceptizer
         
    def initialize(self, x0=None):
        if x0 is None:
            self.generated = nn.Parameter(torch.normal(mean=torch.zeros(28,28), std=torch.ones(28,28)))
        else:
            self.generated = nn.Parameter(x0)
            
    def forward(self):
        activations = self.conceptizer.encode(self.generated.view(1,1,28,28))
        return activations.squeeze()