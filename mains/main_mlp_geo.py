from data_preparation.GeoParser import GeoParser
from dimension_reduction.ordination import ordination2d
import pandas as pd


def adapt_datasets(df1, df2):
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    print("LEN adapted", len(df1.index))
    return df1, df2


def __main__():
    import os

    os.chdir("..") # To return at the root of the project

    from models.discriminative.artificial_neural_networks.MultiLayerPerceptron import MLP
    geo_ids = ["GSE33000"]
    unlabelled_geo_ids = ["GSE33000"]
    load_from_disk = True
    load_merge = False
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    translate = "f"
    extra_class = True # TODO change to put the number... curious to see if more than one is desirable
    meta_destination_folder = "pandas_meta_df"
    plots_folder_path = "/".join([home_path, destination_folder, results_folder, "plots/"])

    dataset_name = "gse33000"
    activation = "relu"
    early_stopping = 200
    n_epochs = 1000
    gt_input = 0
    use_conv = False  # Not applicable if not sequence (images, videos, sentences, DNA...)

    lr = 1e-3
    l1 = 0.
    l2 = 0.
    dropout = 0.5
    batch_size = 32
    is_pruning = False
    # mc = 1
    # iw = 1

    # Neurons layers
    h_dims = [128, 128]

    from utils.utils import adapt_datasets
    g = GeoParser(home_path=home_path, geo_ids=geo_ids)
    g.get_geo(load_from_disk=load_from_disk, automatic_attribute_list=None)
    meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=True)

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
        mlp.run(n_epochs, verbose=2, show_progress=10, hist_epoch=20, is_balanced_relu=False, all0=False)


if __name__ == "__main__":
    __main__()
