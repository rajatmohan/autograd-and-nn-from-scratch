from optim.optimizer import Optimizer
class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.m = [0 for _ in params]  # First moment vector
        self.v = [0 for _ in params]  # Second moment vector
        self.t = 0  # Time step

    def zero_grad(self):
        super().zero_grad()
        self.t = 0
        self.m = [0 for _ in self.params]
        self.v = [0 for _ in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if hasattr(param, 'require_grad') and param.grad is not None:
                # Update biased first and second moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

                # Compute bias-corrected moments estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)