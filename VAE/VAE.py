import torch

import numpy as np

from torch import nn
from torch import optim

from ..Optimizers.config import *

class VAE(nn.Module):
    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.train_loss_v_epoch = None
        self.valid_loss_v_epoch = None
        self.test_loss_v_epoch = None

    def forward(self, x):
        params = self.encoder(x)
        z = self.repametrize(params)
        x_recon = self.decoder(z)

        return x_recon

    def encoder(self, x):
        # This Function needed to be overridden with encoder network
        pass

    def repametrize(self, params):
        # This Function needed to be overridden with the reparametrize trick
        pass

    def decoder(self, z):
        # This Function needed to be overridden with decoder network
        pass

    def loss_fn(self, x, y):
        # This Function needed to be overridden with the appropriate loss
        pass

    def fit(self, train_loader, validation_loader, epochs = 10, optimizer = ADAM, lr = 0.0001, optim_args = dict(), verbose = 1):
        # Function for training
        # train_loader and validation_loader are objects of DataLoader class from PyTorch
        # Depending on the optimizer used required optimizer args which are in PyTorch can be passed
            #   Example: if optmizer = SGD
            #   optim_args = {"momemtum" : 0.01}

        self.train_loss_v_epoch = np.zeros((2, epochs))
        self.valid_loss_v_epoch = np.zeros((2, epochs))


        if optimizer == SGD:
            pass
        elif optimizer == ADA_GRAD:
            pass
        else:
            optimizer = optim.Adam(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print("For epoch {}".format(epoch+1))

            self.train(True)

            train_loss = 0
            count = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, inputs)
                loss.backward()
                optimizer.step()

                curr_loss = loss.item()
                train_loss += curr_loss
                count += 1

                print("\t\tFor Epoch {}, Loss for Batch {} = {}".format(epoch + 1, i + 1, curr_loss))

            avg_train_loss = train_loss/count

            self.train(False)

            valid_loss = 0
            count = 0
            for i, data in enumerate(validation_loader):
                inputs, labels = data
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, inputs)
                valid_loss += loss.item()
                count += 1
            
            avg_valid_loss = valid_loss/count

            print("\tTrain Loss = {},\tValidation Loss = {}".format(avg_train_loss, avg_valid_loss))

            self.train_loss_v_epoch[0][epoch] = epoch
            self.train_loss_v_epoch[0][epoch] = avg_valid_loss

            self.valid_loss_v_epoch[0][epoch] = epoch
            self.valid_loss_v_epoch[0][epoch] = avg_valid_loss

    def generate(self, z):
        # Feed some random z to generate data from decoder

        with torch.no_grad():
            return self.decoder(z)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))