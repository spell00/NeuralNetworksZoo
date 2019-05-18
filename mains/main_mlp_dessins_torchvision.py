import numpy as np
import pandas as pd
from numpy import genfromtxt
import os
import torch
from torchvision import transforms, datasets
from models.discriminative.artificial_neural_networks.MultiLayerPerceptron import MLP

def adapt_datasets(df1, df2):
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    print("LEN adapted", len(df1.index))
    return df1, df2


def __main__():
    train_labels_fname = "train_labels.csv"
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    extra_class = True # TODO change to put the number... curious to see if more than one is desirable

    dataset_name = "dessins"
    n_epochs = 1000

    lr = 1e-3
    l1 = 0.
    l2 = 0.
    batch_size = 256
    resized_shape = 100
    # Neurons layers
    h_dims = [1024, 512, 256, 128]
    input_shape = [1, 100, 100]
    input_size = np.prod([1, resized_shape, resized_shape])

    dir = "data/kaggle_dessins/"
    train_labels = genfromtxt(dir + train_labels_fname, delimiter=",", dtype=str, skip_header=True)[:, 1]
    train_labels_set = set(train_labels)
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop((resized_shape)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
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
    mlp = MLP(input_size=input_size, input_shape=input_shape,
              indices_names=list(range(len(train_labels))), num_classes=len(train_labels_set),
              h_dims=h_dims, extra_class=extra_class, l1=l1, l2=l2, batch_norm=True)

    mlp.x_train = train_ds
    mlp.x_valid = valid_ds
    mlp.x_test = None

    mlp.batch_size = batch_size
    mlp.labels_set = os.listdir('data/kaggle_dessins/train/')
    mlp.num_classes = len(mlp.labels_set)

    mlp.labels_train = train_labels
    mlp.labels = train_labels
    mlp.labels_set = train_labels_set

    mlp.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers", is_unlabelled=False)
    mlp.make_loaders(train_ds=train_ds, valid_ds=valid_ds, test_ds=None, labels_per_class=-1,
                     unlabelled_train_ds=None, unlabelled_samples=True)

    train_total_loss_histories = [[] for x in range(10)]
    train_accuracy_histories = [[] for x in range(10)]
    valid_total_loss_histories = [[] for x in range(10)]
    valid_accuracy_histories = [[] for x in range(10)]
    mlp.train_loader = mlp.train_loader.dataset
    mlp.valid_loader = mlp.valid_loader.dataset
    for i in range(10):
        print("Random train/valid split", i)
        mlp.set_data(labels_per_class=-1, is_example=False, is_split=False, extra_class=extra_class, is_custom_data=True)
        mlp.glorot_init()
        mlp.run(n_epochs, verbose=3, show_progress=10, hist_epoch=20, is_balanced_relu=False, all0=False)


if __name__ == "__main__":
    __main__()
