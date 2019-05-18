from numpy import genfromtxt
from torchvision import transforms, datasets
import torch
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

def __main__():
    from models.semi_supervised.utils.utils import onehot
    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    local_folder = "./data/kaggle_dessins/"
    train_images_fname = "train_images.npy"
    train_labels_fname = "train_labels.csv"
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    dataset_name = "dessins_ssl"
    betas = (0.9, 0.999)
    z_dims = [32]
    a_dims = [50]
    h_dims = [128, 64]
    num_elements = 5
    batch_size = 32
    number_of_flows = 10
    input_shape = [1, 100, 100]
    lr = 3e-5
    warmup = 100
    meta_df = None
    unlabelled_meta_df = None
    use_conv = False
    vae_flavour = "o-sylvester"
    labels_per_class = 100
    early_stopping = 100
    has_unlabelled = False
    load_vae = False
    resume = False # not working... TODO to be corrected, would be useful
    n_epochs = 10000
    auxiliary = True
    mc = 1
    iw = 1
    l1 = 0.
    l2 = 0.

    dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows,a_dim=a_dims[0],
                                       num_elements=num_elements, use_conv=use_conv)

    dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")


    train_labels = genfromtxt("data/kaggle_dessins/train_labels.csv", delimiter=",", dtype=str, skip_header=True)[:, 1]
    train_labels_set = set(train_labels)
    num_classes = len(train_labels_set)
    data_transform = transforms.Compose([
        #transforms.RandomRotation(180),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.ColorJitter()
    ])
    dessins_train_data = datasets.ImageFolder(root='data/kaggle_dessins/train/',
                                              transform=data_transform)
    dessins_valid_data = datasets.ImageFolder(root='data/kaggle_dessins/valid/',
                                              transform=data_transform)
    train_ds = torch.utils.data.DataLoader(dessins_train_data,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
    valid_ds = torch.utils.data.DataLoader(dessins_valid_data,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
    dgm.batch_size = batch_size
    dgm.input_shape = input_shape
    dgm.input_size = np.prod(input_shape)

    dgm.make_loaders(train_ds=train_ds, valid_ds=valid_ds, test_ds=None, labels_per_class=-1,
                     unlabelled_train_ds=None, unlabelled_samples=True)

    dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour)

    dgm.labels_set = os.listdir('data/kaggle_dessins/train/')
    dgm.num_classes = len(dgm.labels_set)
    dgm.labels = train_labels

    dgm.cuda()
    dgm.train_loader = dgm.train_loader.dataset
    dgm.valid_loader = dgm.valid_loader.dataset

    dgm.set_data(labels_per_class=-1, is_example=False, extra_class=False, has_unlabelled_samples=False,
                 is_split=False, is_custom_data=True)
    dgm.set_adgm_layers(is_hebb_layers=False, input_shape=input_shape, h_dims=h_dims)

    dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2, clip_grad=0.)

if __name__ == "__main__":
    __main__()
