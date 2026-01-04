from nn.module import Module
from nn.neuron import Neuron
import random
class Linear(Module):
    def __init__(self, no_inputs, no_outputs, activation=None):
        self.neurons = [Neuron(no_inputs, activation=activation) for _ in range(no_outputs)]
        self.no_inputs = no_inputs
        self.no_outputs = no_outputs
    def forward(self, x):
        if len(x) != self.no_inputs:
            raise ValueError(f"Input size {len(x)} does not match expected size {self.no_inputs}")
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs
