import math
import numpy as np
import time

import scipy.sparse

from optimizer import StochasticOptimizer


def nnz(x):
    if scipy.sparse.issparse(x):
        return x.nnz
    return np.count_nonzero(x)


class Sgd(StochasticOptimizer):
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
        avoid_cache_miss (bool, optional): whether to make iterations faster by using chunks of the data
            that are adjacent to each other. Implemented by sampling an index and then using the next
            batch_size samples to obtain the gradient. May lead to slower iteration convergence (default: True)
    """
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None,
                 batch_size=1, avoid_cache_miss=True, importance_sampling=False, *args, **kwargs):
        super(Sgd, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size
        self.avoid_cache_miss = avoid_cache_miss
        self.importance_sampling = importance_sampling
        
    def step(self):
        t1 = time.perf_counter()
        if self.avoid_cache_miss:
            i = np.random.choice(self.loss.n)
            idx = np.arange(i, i + self.batch_size)
            idx %= self.loss.n
            self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        else:
            self.grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size, importance_sampling=self.importance_sampling)
        self.t_grad += time.perf_counter() - t1
        self.coords_grad += nnz(self.grad)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
        t2 = time.perf_counter()
        if self.use_prox:
            self.coords_prox += nnz(self.x)
            self.x = self.loss.regularizer.prox(self.x, self.lr)
        self.t_prox += time.perf_counter() - t2
    
    def init_run(self, *args, **kwargs):
        super(Sgd, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        self.t_prox = 0
        self.t_grad = 0
        self.coords_grad = 0
        self.coords_prox = 0
        self.trace.coords_grad = []
        self.trace.coords_prox = []
        self.trace.coords = []
        
    def update_trace(self, first_iterations=10):
        super(Sgd, self).update_trace()
        self.trace.coords_grad.append(self.coords_grad)
        self.trace.coords_prox.append(self.coords_prox)
        self.trace.coords.append(self.coords_grad + self.coords_prox)


class Shuffling(StochasticOptimizer):
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
    def __init__(self, reshuffle=False, prox_every_it=False, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, epoch_start_decay=1, batch_size=1, importance_sampling=False, *args, **kwargs):
        super(Shuffling, self).__init__(*args, **kwargs)
        self.reshuffle = reshuffle
        self.prox_every_it = prox_every_it
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.epoch_start_decay = epoch_start_decay
        self.batch_size = batch_size
        self.importance_sampling = importance_sampling
        
        self.steps_per_epoch = math.ceil(self.loss.n/batch_size)
        self.epoch_max = self.it_max // self.steps_per_epoch
        if epoch_start_decay is None and np.isfinite(self.epoch_max):
            self.epoch_start_decay = 1 + self.epoch_max // 40
        elif epoch_start_decay is None:
            self.epoch_start_decay = 1
        if importance_sampling:
            self.sample_counts = self.individ_smoothness / np.mean(self.individ_smoothness)
            self.sample_counts = np.int64(np.ceil(self.sample_counts))
            self.idx_with_copies = np.repeat(np.arange(self.loss.n), self.sample_counts)
            self.n_copies = sum(self.sample_counts)
            self.steps_per_epoch = math.ceil(self.n_copies / batch_size)
        
    def step(self):
        t1 = time.perf_counter()
        if self.it%self.steps_per_epoch == 0:
            # Start new epoch
            if self.reshuffle:
                if not self.importance_sampling:
                    self.permutation = np.random.permutation(self.loss.n)
                else:
                    self.permutation = np.random.permutation(self.idx_with_copies)
                self.sampled_permutations += 1
            self.i = 0
        idx_perm = np.arange(self.i, min(len(self.permutation), self.i+self.batch_size))
        idx = self.permutation[idx_perm]
        self.i += self.batch_size
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        if not self.importance_sampling:
            normalization = self.loss.n / self.steps_per_epoch
        else:
            normalization = self.n_copies / self.steps_per_epoch * self.sample_counts[idx]
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)
        self.t_grad += time.perf_counter() - t1
        self.coords_grad += nnz(self.grad)
        denom_const = 1 / self.lr0
        it_decrease = self.steps_per_epoch * max(0, self.finished_epochs-self.epoch_start_decay)
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*it_decrease**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad
        end_of_epoch = self.it%self.steps_per_epoch == self.steps_per_epoch-1
        if end_of_epoch and self.use_prox:
            self.coords_prox += nnz(self.x)
            self.x = self.loss.regularizer.prox(self.x, self.lr * self.steps_per_epoch)
            self.finished_epochs += 1
    
    def init_run(self, *args, **kwargs):
        super(Shuffling, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        self.finished_epochs = 0
        self.permutation = np.random.permutation(self.loss.n)
        self.sampled_permutations = 1
        self.t_prox = 0
        self.t_grad = 0
        self.coords_grad = 0
        self.coords_prox = 0
        self.trace.coords_grad = []
        self.trace.coords_prox = []
        self.trace.coords = []
        
    def update_trace(self, first_iterations=10):
        super(Shuffling, self).update_trace()
        self.trace.coords_grad.append(self.coords_grad)
        self.trace.coords_prox.append(self.coords_prox)
        self.trace.coords.append(self.coords_grad + self.coords_prox)
