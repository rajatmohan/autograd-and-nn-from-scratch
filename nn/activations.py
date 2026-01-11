from nn.module import Module

class ReLU(Module):
    def forward(self, x):
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        return x.relu()

class Tanh(Module):
    def forward(self, x):
        if isinstance(x, list):
            return [xi.tanh() for xi in x]
        return x.tanh()


