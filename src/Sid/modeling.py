# Standard Imports
from __future__ import division
from __future__ import print_function

# Third Party Imports
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models

# Local Imports
from src.load_data import CustomDataset, load_and_clean_data


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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """

    :param model_name:
    :param num_classes:
    :param feature_extract:
    :param use_pretrained:
    :return:
    """

    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_Available() else "cpu")
    n_epochs = 50
    train_batch_size = 8
    test_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-4
    momentum = 0.5
    log_interval = 10
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Define the train and test data loader
    train_data, test_data = load_and_clean_data(path="../../data/data.h5")
    train_dataset = CustomDataset(train_data, split="train")
    test_dataset = CustomDataset(test_data, split="test")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

    # load the model
    model_name = "alexnet"
    num_classes = 3
    feature_extract = False
    model_ft, input_size = initialize_model(model_name, num_classes,
                                            feature_extract, use_pretrained=True)
    model_ft.to(device)

    # define the optimizer
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer = optim.AdamW(params_to_update, lr=learning_rate,
                            weight_decay=weight_decay)

    # define the criterion


    # train
