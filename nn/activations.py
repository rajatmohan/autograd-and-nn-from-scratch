from nn.module import Module

class ReLU(Module):
    def forward(self, x):
        if isinstance(x, list):
            return x[0].relu() if len(x) == 1 else [xi.relu() for xi in x]
        return x.relu()

class Tanh(Module):
    def forward(self, x):
        if isinstance(x, list):
            return x[0].tanh() if len(x) == 1 else [xi.tanh() for xi in x]
        return x.tanh()

class Sigmoid(Module):
    def forward(self, x):
        if isinstance(x, list):
            return x[0].sigmoid() if len(x) == 1 else [xi.sigmoid() for xi in x]
        return x.sigmoid()

