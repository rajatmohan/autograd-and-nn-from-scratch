from core.myTensor import MyTensor
from loss.loss import Loss

class BCELoss(Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        
    def __call__(self, y_pred, y_true):
        epsilon = 1e-12  # small constant to avoid log(0)
        if isinstance(y_pred, MyTensor):
            return - ((y_true * (y_pred.log()) + (1 - y_true) * ((1 - y_pred).log())))
        assert len(y_pred) == len(y_true)
        losses = [-(yt*((yp + epsilon).log()) + (1 - yt)*((1 - yp + epsilon).log())) for yp, yt in zip(y_pred, y_true)]
        return self.reduce(losses)