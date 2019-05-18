import numpy as np
import pandas as pd
from numpy import genfromtxt

from models.discriminative.artificial_neural_networks.MultiLayerPerceptron import MLP

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
    activation = "relu"
    early_stopping = 200
    n_epochs = 1000
    gt_input = 0
    use_conv = False  # Not applicable if not sequence (images, videos, sentences, DNA...)

    lr = 1e-5
    l1 = 1e-5
    l2 = 1e-10
    dropout = 0.5
    batch_size = 16
    is_pruning = False
    # mc = 1
    # iw = 1

    # Neurons layers
    h_dims = [1024, 1024, 1024]

    from utils.utils import adapt_datasets
    train_arrays = np.load(local_folder + train_images_fname, encoding="latin1")
    train_dataset = np.vstack(train_arrays[:, 1])
    train_labels = genfromtxt(local_folder + train_labels_fname, delimiter=",", dtype=str, skip_header=True)[:, 1]
    test_dataset = np.vstack(np.load(local_folder + "test_images.npy", encoding="latin1")[:, 1])

    meta_df = pd.DataFrame(train_dataset, columns=train_labels)
    img_shape = [1, 100, 100]
    labels = set(list(meta_df.columns))

    mlp = MLP(input_size=meta_df.shape[0], input_shape=(meta_df.shape[0]),
              indices_names=list(range(meta_df.shape[0])), num_classes=len(labels),
              h_dims=h_dims, extra_class=extra_class, l1=l1, l2=l2, batch_norm=True)

    mlp.labels = labels
    mlp.labels_set = list(set(labels))

    mlp.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers", is_unlabelled=False)
    print("Labeled data shape (35371, 624)", meta_df.shape)
    if meta_df is not None:
        mlp.import_dataframe(meta_df, batch_size, labelled=True)

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
