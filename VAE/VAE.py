import torch

import numpy as np

from torch import nn
from torch import optim

from ..AutoEncoders import AutoEncoder
from ..utils.learning_types import SELF_SUPERVISED
from ..Optimizers.config import *

class VAE(AutoEncoder):
    def __init__(self) -> None:
        super(VAE, self).__init__()

    def forward(self, x):
        params = self.encoder(x)
        z = self.repametrize(params)
        x_recon = self.decoder(z)

        return x_recon

    def repametrize(self, params):
        # This Function needed to be overridden with the reparametrize trick
        pass

    def kl(self):
        # Overload this function to define loss by KL divergence
        pass

    def recon_loss(self, x, y):
        # Overload this function for recon_loss
        # Defaulted by MSE
        return ((x-y)**2).sum()

    def loss_fn(self, x, y):
        return self.kl() - self.recon_loss(x, y)

    def generate(self, z):
        # Feed some random z to generate data from decoder

        with torch.no_grad():
            return self.decoder(z)

class Beta_VAE(VAE):
    def __init__(self, beta = 10) -> None:
        super(Beta_VAE, self).__init__()

        self.beta = beta

    def loss_fn(self, x, y):
        return self.beta * self.kl() - self.recon_loss(x, y)

    def set_beta(self, beta):
        self.beta = beta