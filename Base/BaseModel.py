import torch

import numpy as np

from torch import nn
from torch import optim

from pyML.utils.learning_types import SELF_SUPERVISED, SUPERVISED

from ..Optimizers.config import *

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

        self.train_loss_v_epoch = None
        self.valid_loss_v_epoch = None
        self.test_loss_v_epoch = None

    def forward(self, x):
        pass

    def loss_fn(self, x, y):
        pass

    def fit(self, train_loader, validation_loader = None, epochs = 10, optimizer = ADAM, lr = 0.0001, optim_args = dict(), learning_type = SUPERVISED, verbose = 1):
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
                
                if learning_type == SUPERVISED:
                    loss = self.loss_fn(outputs, labels)
                elif learning_type == SELF_SUPERVISED:
                    loss = self.loss_fn(outputs, inputs)

                loss.backward()
                optimizer.step()

                curr_loss = loss.item()
                train_loss += curr_loss
                count += 1

                print("\t\tFor Epoch {}, Loss for Batch {} = {}".format(epoch + 1, i + 1, curr_loss))

            avg_train_loss = train_loss/count

            self.train(False)

            if validation_loader:
                valid_loss = 0
                count = 0
                for i, data in enumerate(validation_loader):
                    inputs, labels = data
                    outputs = self.forward(inputs)
                    
                    if learning_type == SUPERVISED:
                        loss = self.loss_fn(outputs, labels)
                    elif learning_type == SELF_SUPERVISED:
                        loss = self.loss_fn(outputs, inputs)

                    valid_loss += loss.item()
                    count += 1
            
                avg_valid_loss = valid_loss/count

                print("\tTrain Loss = {},\tValidation Loss = {}".format(avg_train_loss, avg_valid_loss))

                self.valid_loss_v_epoch[0][epoch] = epoch
                self.valid_loss_v_epoch[0][epoch] = avg_valid_loss
            else:
                print("\tTrain Loss = {}".format(avg_train_loss))

            self.train_loss_v_epoch[0][epoch] = epoch
            self.train_loss_v_epoch[0][epoch] = avg_train_loss

    def predict(self, x):
        with torch.no_grad:
            return self.forward(x)

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def load_state(self, path):
        self.load_state_dict(torch.load(path))