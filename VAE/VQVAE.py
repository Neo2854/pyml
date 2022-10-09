import torch

import numpy as np

from torch import nn
from torch import optim

from ..AutoEncoders import AutoEncoder
from ..utils.learning_types import SELF_SUPERVISED
from ..Optimizers.config import *
from .VAE import VAE

class VQVAE(VAE):
    def __init__(self, device=None) -> None:
        super(VAE, self).__init__(device)