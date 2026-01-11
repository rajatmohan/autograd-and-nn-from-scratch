from optim.optimizer import Optimizer
class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            if hasattr(param, 'require_grad') and param.grad is not None:
                param.data -= self.lr * param.grad