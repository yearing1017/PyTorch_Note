import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch/self.max_iter) ** self.power
                for base_lr in self.base_lrs]


optimizer = optim.Adam(danet_model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer_lr_scheduler = PolyLR(optimizer, max_iter=40, power=0.9)
