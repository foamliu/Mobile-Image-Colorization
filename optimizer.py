class MICOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0
        self.lr = 3e-5

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        # Initial learning rate was 3 × 10−5 and dropped to 10−5
        # and 3 × 10−6 when loss plateaued, at 200k and 375k iterations, respectively
        if self.step_num == 200000:
            self.lr = 1e-5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        elif self.step_num == 375000:
            self.lr = 3e-6
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def clip_gradient(self, grad_clip):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
