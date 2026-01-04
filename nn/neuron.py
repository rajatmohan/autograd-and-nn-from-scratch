from nn.module import Module
from core.myTensor import MyTensor
import random
class Neuron(Module):
    def __init__(self, no_inputs, activation='tanh'):
        self.w = [MyTensor(random.uniform(-1, 1), require_grad = True) for _ in range(no_inputs)]
        self.b = MyTensor(random.uniform(-1, 1), require_grad = True)
        self.activation = activation
    
    def forward(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"Input size {len(x)} does not match number of weights {len(self.w)}")
        preact = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if(self.activation == 'tanh'):
            return preact.tanh()
        elif(self.activation == 'relu'):
            return preact.relu()
        return preact


