from torch import nn
from torch import optim

from ..Base.BaseModel import BaseModel
from ..utils.learning_types import SELF_SUPERVISED
from ..Optimizers.config import *

class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()

        self.train_loss_v_epoch = None
        self.valid_loss_v_epoch = None
        self.test_loss_v_epoch = None

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def encoder(self,x):
        pass

    def decoder(self, x):
        pass

    def loss_fn(self, x, y):
        # Overload this function for loss
        # Defaulted by MSE
        return ((x-y)**2).sum()

    def fit(self, train_loader, validation_loader = None, epochs = 10, optimizer = ADAM, lr = 0.0001, optim_args = dict(), verbose = 1):
        # Function for training
        # train_loader and validation_loader are objects of DataLoader class from PyTorch
        # Depending on the optimizer used required optimizer args which are in PyTorch can be passed
            #   Example: if optmizer = SGD
            #   optim_args = {"momemtum" : 0.01}

        super(BaseModel, self).fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=epochs,
            optimizer=optimizer,
            lr=lr,
            optim_args=optim_args,
            learning_type=SELF_SUPERVISED,
            verbose=verbose
        )