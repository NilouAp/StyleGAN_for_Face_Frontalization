import math
import torch
from torch.optim.optimizer import Optimizer


class Ranger(Optimizer):

	def __init__(self, params, lr=1e-3,  # lr
				 alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
				 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
				 use_gc=True, gc_conv_only=False
				 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
				 ):

		# parameter checks
		if not 0.0 <= alpha <= 1.0:
			raise ValueError(f'Invalid slow update rate: {alpha}')
		if not 1 <= k:
			raise ValueError(f'Invalid lookahead steps: {k}')
		if not lr > 0:
			raise ValueError(f'Invalid Learning Rate: {lr}')
		if not eps > 0:
			raise ValueError(f'Invalid eps: {eps}')

		# prep defaults and init torch.optim base
		defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,
						eps=eps, weight_decay=weight_decay)
		super().__init__(params, defaults)

		# adjustable threshold
		self.N_sma_threshhold = N_sma_threshhold

		# look ahead params

		self.alpha = alpha
		self.k = k

		# radam buffer for state
		self.radam_buffer = [[None, None, None] for ind in range(10)]

		# gc on or off
		self.use_gc = use_gc

		# level of gradient centralization
		self.gc_gradient_threshold = 3 if gc_conv_only else 1

	def __setstate__(self, state):
		super(Ranger, self).__setstate__(state)