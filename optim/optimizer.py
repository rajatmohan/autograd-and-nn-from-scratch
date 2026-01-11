class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self):
        raise NotImplementedError