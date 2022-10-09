import torch

import numpy as np

from torch import nn
from torch import optim

from ..AutoEncoders import AutoEncoder
from ..utils.learning_types import SELF_SUPERVISED
from ..Optimizers.config import *
from .VAE import VAE

class BetaVAE(VAE):
    def __init__(self, beta = 10, device = None) -> None:
        super(BetaVAE, self).__init__(device=device)

        self.beta = beta

    def loss_fn(self, x, y):
        return self.beta * self.kl() - self.recon_loss(x, y)

    def set_beta(self, beta):
        self.beta = beta