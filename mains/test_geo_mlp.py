def main():
    from data_preparation.GeoParser import GeoParser
    from models.discriminative.artificial_neural_networks.MultiLayerPerceptron import MLP
    load_from_disk = True
    load_merge = False

    geo_ids = ["GSE33000"]
    # files_destinations
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    meta_destination_folder = "pandas_meta_df"

    plots_folder_path = "/".join([home_path, destination_folder, results_folder, "plots/"])
    dataset_name = "gse33000_no_huntington"
    activation = "relu"
    # nrep = 3
    early_stopping = 200
    n_epochs = 1000
    gt_input = 0
    extra_class = False
    dataset_name = dataset_name + "extra_class"+str(extra_class)
    # if ladder is yes builds a ladder vae. Do not combine with auxiliary (yet; might be possible and relatively
    # not too hard to implement, but might be overkill. Might be interesting too)
    translate = "n"

    use_conv = False  # Not applicable if not sequence (images, videos, sentences, DNA...)
    lr = 1e-4
    l1 = 0.
    l2 = 0.
    batch_size = 32
    # mc = 1
    # iw = 1

    # Neurons layers
    h_dims = [128, 128]
    from utils.utils import adapt_datasets
    g = GeoParser(home_path=home_path, geo_ids=geo_ids)
    g.get_geo(load_from_disk=load_from_disk, automatic_attribute_list=None)
    meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=True)
    if translate is "y":
        for geo_id in geo_ids:
            g.translate_indices_df(geo_id, labelled=True)
    labels = set(list(meta_df.columns))
    print(labels)
    mlp = MLP(input_size=meta_df.shape[0], input_shape=(meta_df.shape[0]),
              indices_names=list(range(meta_df.shape[0])), num_classes=len(labels),
              h_dims=h_dims, extra_class=extra_class, l1=l1, l2=l2, batch_norm=True)

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
    for i in range(100):
        print("Random train/valid split", i)
        mlp.set_data(labels_per_class=-1, is_example=False, extra_class=extra_class, ignore_training_inputs=1)
        mlp.glorot_init()
        mlp.run(n_epochs, verbose=0, show_progress=10, hist_epoch=20, is_balanced_relu=True, all0=True,
               overall_mean=True)


if __name__ == "__main__":
    main()
