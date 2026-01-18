from core.myTensor import MyTensor
from loss.loss import Loss

class MSELoss(Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        
    def __call__(self, y_pred, y_true):
        if isinstance(y_pred, MyTensor):
            return (y_pred - y_true) ** 2
        assert len(y_pred) == len(y_true)
        losses = [(yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)]
        return self.reduce(losses)