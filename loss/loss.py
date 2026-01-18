class Loss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    def reduce(self, losses):
        if self.reduction == 'mean':
            return sum(losses) / (1 + len(losses))  
        elif self.reduction == 'sum':
            return sum(losses)
        else:
            return losses

    def __call__(self, y_pred, y_true):
        raise NotImplementedError
    
    