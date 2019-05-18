import numpy as np
import pandas as pd
from numpy import genfromtxt
import numpy as np
from numpy import genfromtxt
import torch
from torchvision import transforms, datasets
from models.discriminative.artificial_neural_networks.ConvNet import ConvNet
from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
import os


def adapt_datasets(df1, df2):
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    print("LEN adapted", len(df1.index))
    return df1, df2


def __main__():
    local_folder = "./data/kaggle_dessins/"
    train_images_fname = "train_images.npy"
    train_labels_fname = "train_labels.csv"
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    extra_class = True # TODO change to put the number... curious to see if more than one is desirable
    meta_destination_folder = "pandas_meta_df"
    plots_folder_path = "/".join([home_path, destination_folder, results_folder, "plots/"])

    dataset_name = "dessins"
    vae_flavour = "o-sylvester"
    activation = "relu"
    early_stopping = 200
    n_epochs = 1000
    gt_input = 0
    use_conv = False  # Not applicable if not sequence (images, videos, sentences, DNA...)

    lr = 1e-5
    l1 = 0.
    l2 = 0.
    dropout = 0.5
    batch_size = 64
    number_of_flows = 4
    num_elements = 3
    a_dim = 20
    lr = 1e-4
    z_dims = [50]
    is_pruning = False
    # mc = 1
    # iw = 1

    # Neurons layers
    h_dims = [1024]
    planes = [1, 16, 32, 64, 128, 256, 512]
    kernels = [3, 3, 3, 3, 3, 3]
    pooling_layers = [1, 1, 1, 1, 1, 1]

    train_labels = genfromtxt("data/kaggle_dessins/" + train_labels_fname, delimiter=",", dtype=str, skip_header=True)[
                   :, 1]
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
    dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, a_dim=a_dim,
                                       num_elements=num_elements, use_conv=use_conv)

    dgm.batch_size = batch_size
    dgm.input_shape = [1, 100, 100]
    dgm.input_size = 10000
    dgm.labels_set = os.listdir('data/kaggle_dessins/train/')
    dgm.num_classes = len(dgm.labels_set)


    dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    dgm.make_loaders(train_ds=train_ds, valid_ds=valid_ds, test_ds=None, labels_per_class=labels_per_class,
                     unlabelled_train_ds=None, unlabelled_samples=True)

    input_shape = [1, 100, 100]
    labels = train_labels

    mlp = ConvNet(input_shape=input_shape, num_classes=len(labels),
                  h_dims=h_dims, extra_class=extra_class, l1=l1, l2=l2, batch_norm=True)

    mlp.labels = labels
    mlp.labels_set = list(set(labels))

    mlp.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers", is_unlabelled=False)

    train_total_loss_histories = [[] for x in range(10)]
    train_accuracy_histories = [[] for x in range(10)]
    valid_total_loss_histories = [[] for x in range(10)]
    valid_accuracy_histories = [[] for x in range(10)]
    for i in range(10):
        print("Random train/valid split", i)
        mlp.set_data(labels_per_class=-1, is_example=False, extra_class=extra_class)
        mlp.glorot_init()
        mlp.run(n_epochs, verbose=3, show_progress=10, hist_epoch=20, is_balanced_relu=False, all0=False)


if __name__ == "__main__":
    __main__()
