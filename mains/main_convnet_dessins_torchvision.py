import numpy as np
import pandas as pd
from numpy import genfromtxt
import os
import torch
from torchvision import transforms, datasets
from models.discriminative.artificial_neural_networks.ConvNet import ConvNet

def adapt_datasets(df1, df2):
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    print("LEN adapted", len(df1.index))
    return df1, df2


def __main__():
    train_labs = "train_labels.csv"
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    extra_class = True # TODO change to put the number... curious to see if more than one is desirable

    dataset_name = "dessins"
    n_epochs = 10000

    lr = 1e-3
    l1 = 1e-8
    l2 = 1e-8
    batch_size = 64

    # Neurons layers
    h_dims = [1024]
    input_shape = [1, 100, 100]
    planes = [1, 16, 32, 64, 128, 256, 512, 1024]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3]
    pooling_layers = [True, True, True, True, True, True, False, False]

    train_labels = genfromtxt("data/kaggle_dessins/" + train_labs, delimiter=",", dtype=str, skip_header=True)[:, 1]
    train_labels_set = set(train_labels)
    num_classes = len(train_labels_set)
    train_transform = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.], std=[1.]), # TODO NECESSAIRE?
        transforms.ColorJitter()
    ])
    valid_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.], std=[1.])  # TODO NECESSAIRE?
    ])

    dessins_train_data = datasets.ImageFolder(root='data/kaggle_dessins/train/',
                                              transform=train_transform)
    dessins_valid_data = datasets.ImageFolder(root='data/kaggle_dessins/valid/',
                                              transform=valid_transform)
    train_ds = torch.utils.data.DataLoader(dessins_train_data,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
    valid_ds = torch.utils.data.DataLoader(dessins_valid_data,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
    cnn = ConvNet(input_shape=input_shape, indices_names=list(range(len(train_labels))), num_classes=num_classes,  h_dims=h_dims, planes=planes,
                  kernels=kernels, pooling_layers=pooling_layers, extra_class=extra_class,  l1=l1, l2=l2,
                  batch_norm=True)
    cnn.x_train = train_ds
    cnn.x_valid = valid_ds
    cnn.x_test = None

    cnn.batch_size = batch_size
    cnn.labels_set = os.listdir('data/kaggle_dessins/train/')
    cnn.num_classes = len(cnn.labels_set)

    cnn.labels_train = train_labels
    cnn.labels = train_labels
    cnn.labels_set = train_labels_set

    cnn.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers", is_unlabelled=False)
    cnn.make_loaders(train_ds=train_ds, valid_ds=valid_ds, test_ds=None, labels_per_class=-1,
                     unlabelled_train_ds=None, unlabelled_samples=True)

    cnn.train_loader = cnn.train_loader.dataset
    cnn.valid_loader = cnn.valid_loader.dataset
    cnn.set_data(labels_per_class=-1, is_example=False, is_split=False, extra_class=extra_class, is_custom_data=True)
    cnn.glorot_init()
    cnn.run(n_epochs, verbose=1, show_progress=10, hist_epoch=20, is_balanced_relu=False, all0=False)
    cnn.save_model() # Should be saved at the end of each epoch, but just to be sure

if __name__ == "__main__":
    __main__()
