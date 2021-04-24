# Local Imports
from src.load_data import CustomDataset

# Third Party Imports
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn


class RADARModel(nn.Module):

    def __init__(self, config):
        super(RADARModel, self).__init__()

        self.config = config
        self.encoder = None
        self.decoder = None

    def forward(self, image):
        """
        Forward Propagation
        :param image: a tensor of dimensions (N, 256, 32)
        :return:
        """

        return None

    def custom_loss_fn(self):
        return None

    if __name__ == '__main__':
        device = torch.device("cuda" if torch.cuda.is_Available() else "cpu")

        n_epochs = 10
        train_batch_Size = 128
        test_batch_size = 128
        learning_rate = 0.01
        momentum = 0.5
        log_interval = 10

        random_seed = 42
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)

        # Define the train and test data loader
        complete_data_set = CustomDataset("../../data/Data.h5")


