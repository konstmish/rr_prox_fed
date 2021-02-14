import math
import numpy as np
import psutil
import ray
import time

from optimizer import StochasticOptimizer


@ray.remote
def local_step(x, lr, loss, it_local, batch_size):
    x_local = x * 1.
    for i in range(it_local):
        x_local -= lr * loss.stochastic_gradient(x_local, batch_size=batch_size)
    return x_local


@ray.remote
def local_epoch(x, lr, loss, batch_size):
    permutation = np.random.permutation(loss.n)
    x_local = x * 1.
    i = 0
    while i < loss.n:
        i_max = min(loss.n, i + batch_size)
        idx = permutation[i:i_max]
        x_local -= lr * loss.stochastic_gradient(x_local, idx=idx)
        i += batch_size
    return x_local


class LocalSgd(StochasticOptimizer):
    """
    Stochastic gradient descent with decreasing or constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        lr_decay_coef (float, optional): the coefficient in front of the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value
            is mu/2, where mu is the strong convexity constant
        lr_decay_power (float, optional): the power to exponentiate the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value is 1 (default: 1)
        it_start_decay (int, optional): how many iterations the step-size is kept constant
            By default, will be set to have about 2.5% of iterations with the step-size equal to lr0
        batch_size (int, optional): the number of samples from the function to be used at each iteration
    """
    def __init__(self, it_local, n_workers=None, iid=False, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None,
                 batch_size=1, *args, **kwargs):
        super(LocalSgd, self).__init__(*args, **kwargs)
        self.it_local = it_local
        if n_workers is None:
            n_workers = psutil.cpu_count(logical=False)
        self.n_workers = n_workers
        self.iid = iid
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size
        
    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local*self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.x = np.mean(ray.get([local_step.remote(x_id, lr_decayed, loss_id, self.it_local, self.batch_size) for i in range(self.n_workers)]), axis=0)
        else:
            pass
    
    def init_run(self, *args, **kwargs):
        super(LocalSgd, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if not self.iid:
            raise ValueError('Blah')
        
    def update_trace(self, first_iterations=10):
        super(LocalSgd, self).update_trace()


class LocalShuffling(StochasticOptimizer):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.
    For a formal description and convergence guarantees, see
        https://arxiv.org/abs/2006.05988
    
    The method is sensitive to finishing the final epoch, so it will terminate earlier 
    than it_max if it_max is not divisible by the number of steps per epoch.
    
    Arguments:
        reshuffle (bool, optional): whether to get a new permuation for every new epoch.
            For convex problems, only a single permutation should suffice and it can run faster (default: False)
        prox_every_it (bool, optional): whether to use proximal operation every iteration 
            or only at the end of an epoch. Theory supports the latter. Only used if the loss includes
            a proximal regularizer (default: False)
        lr0 (float, optional): an estimate of the inverse smoothness constant, this step-size
            is used for the first epoch_start_decay epochs. If not given, it will be set
            with the value in the loss.
        lr_max (float, optional): a maximal step-size never to be exceeded (default: np.inf)
        lr_decay_coef (float, optional): the coefficient in front of the number of finished epochs
            in the denominator of step-size. For strongly convex problems, a good value
            is mu/3, where mu is the strong convexity constant
        lr_decay_power (float, optional): the power to exponentiate the number of finished epochs
            in the denominator of step-size. For strongly convex problems, a good value is 1 (default: 1)
        epoch_start_decay (int, optional): how many epochs the step-size is kept constant
            By default, will be set to have about 2.5% of iterations with the step-size equal to lr0
        batch_size (int, optional): the number of samples from the function to be used at each iteration
    """
    def __init__(self, n_workers=None, iid=False, reshuffle=False, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, epoch_start_decay=1, batch_size=1, *args, **kwargs):
        super(LocalShuffling, self).__init__(*args, **kwargs)
        if n_workers is None:
            n_workers = psutil.cpu_count(logical=False)
        self.n_workers = n_workers
        self.iid = iid
        self.reshuffle = reshuffle
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.epoch_start_decay = epoch_start_decay
        self.batch_size = batch_size
        
        self.steps_per_epoch = math.ceil(self.loss.n/batch_size)
        self.epoch_max = self.it_max // self.steps_per_epoch
        if epoch_start_decay is None and np.isfinite(self.epoch_max):
            self.epoch_start_decay = 1 + self.epoch_max // 40
        elif epoch_start_decay is None:
            self.epoch_start_decay = 1
        
    def step(self):
        if self.it%self.steps_per_epoch == 0 and self.reshuffle:
            # Start new epoch
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0
            self.sampled_permutations += 1
        idx_perm = np.arange(self.i, min(self.loss.n, self.i+self.batch_size))
        idx = self.permutation[idx_perm]
        self.i += self.batch_size
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        normalization = self.loss.n / self.steps_per_epoch
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.loss.n/self.batch_size*self.lr_decay_coef*max(0, self.finished_epochs-self.epoch_start_decay)**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.x = np.mean(ray.get([local_epoch.remote(x_id, lr_decayed, loss_id, batch_size=self.batch_size) for i in range(self.n_workers)]), axis=0)
        else:
            pass
        self.finished_epochs += 1
        
    
    def init_run(self, *args, **kwargs):
        super(LocalShuffling, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        self.finished_epochs = 0
        self.permutation = np.random.permutation(self.loss.n)
        self.sampled_permutations = 1
        
    def update_trace(self, first_iterations=10):
        super(LocalShuffling, self).update_trace()
