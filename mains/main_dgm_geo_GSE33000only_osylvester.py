from data_preparation.GeoParser import GeoParser
import pandas as pd


def adapt_datasets(df1, df2):
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    print("LEN adapted", len(df1.index))
    return df1, df2


def print_infos():
    pass


def get_num_parameters():
    pass


def __main__():
    geo_ids = ["GSE33000"]
    unlabelled_geo_ids = ["GSE33000"]
    bad_geo_ids = []
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    dataset_name = "mnist_ssl"
    h_dims = [512, 256]
    h_dims_classifier = [128]
    betas = (0.9, 0.999)
    z_dims = [100]
    a_dim = [100]
    num_elements = 3
    batch_size = 64 # if smaller not working... should work though...
    number_of_flows = 10
    input_shape = [1, 35371]
    # Total number of inputs in microarray. Each microarray might have a different number,
    # corresponding in part often, but different microarrays might be difficult to put
    # together if they don't sufficiently overlap (though not impossible... )
    warmup = 0
    use_conv = False
    vae_flavour = "o-sylvester"
    labels_per_class = -1
    early_stopping = 100
    has_unlabelled = False
    load_vae = False
    resume = False # not working... TODO to be corrected, would be useful
    n_epochs = 10000
    auxiliary = True
    ladder = False
    mc = 1
    iw = 1
    l1 = 1e-6
    l2 = 1e-6
    lr = 1e-5
    load_from_disk = True
    translate = False
    load_merge = False
    num_classes = 3

    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    from models.semi_supervised.deep_generative_models.models.ladder_dgm import LadderDeepGenerativeModel
    from models.semi_supervised.deep_generative_models.models.dgm import DeepGenerativeModel
    g = GeoParser(home_path=home_path, geo_ids=geo_ids, unlabelled_geo_ids=unlabelled_geo_ids, bad_geo_ids=bad_geo_ids)
    g.get_geo(load_from_disk=load_from_disk, automatic_attribute_list=True)
    meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=True)
    unlabelled_meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=False)
    if translate is "y":
        for geo_id in geo_ids:
            g.translate_indices_df(geo_id, labelled=True)
        for geo_id in unlabelled_geo_ids:
            g.translate_indices_df(geo_id, labelled=False)
    meta_df, unlabelled_meta_df = adapt_datasets(meta_df, unlabelled_meta_df)

    is_example = False
    extra_class = True

    if auxiliary:
        dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, a_dim=a_dim,
                                           num_elements=num_elements, use_conv=use_conv,
                                           labels_per_class=labels_per_class)

        dgm.set_configs(num_classes=num_classes, extra_class=extra_class, home_path=home_path, results_folder=results_folder,
                        data_folder=data_folder, destination_folder=destination_folder, dataset_name=dataset_name,
                        lr=lr, meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    elif ladder:
        dgm = LadderDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, auxiliary=False,
                                        labels_per_class=labels_per_class)

        dgm.set_configs(extra_class=extra_class, home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                        destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                        meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")
    else:
        print(vae_flavour)
        dgm = DeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, a_dim=None, auxiliary=False,
                                  num_elements=num_elements, labels_per_class=labels_per_class)

        dgm.set_configs(extra_class=extra_class, home_path=home_path, results_folder=results_folder,
                        data_folder=data_folder, destination_folder=destination_folder, dataset_name=dataset_name,
                        lr=lr, meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    if auxiliary:
        if use_conv:
            dgm.set_conv_adgm_layers(input_shape=input_shape)
        else:
            dgm.set_adgm_layers(h_dims=h_dims_classifier, input_shape=input_shape)
    elif ladder:
        dgm.set_ldgm_layers()
    else:
        if use_conv:
            dgm.set_conv_dgm_layers(input_shape=input_shape)
        else:
            dgm.set_dgm_layers(input_shape=input_shape)

    # import the M1 in the M1+M2 model (Kingma et al, 2014)
    if load_vae:
        print("Importing the model: ", dgm.model_file_name)
        if use_conv:
            dgm.import_cvae()
        else:
            dgm.load_ae(load_history=False)
        #dgm.set_dgm_layers_pretrained()

    if resume:
        print("Resuming training")
        dgm.load_model()
    dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour)

    if meta_df is not None:
        dgm.import_dataframe(meta_df, batch_size, labelled=True)
        if has_unlabelled:
            dgm.import_dataframe(unlabelled_meta_df, batch_size, labelled=False)
        else:
            dgm.import_dataframe(meta_df, batch_size, labelled=False)
    dgm.set_data(labels_per_class, ratio_training=0.8, ratio_valid=0.1, is_example=is_example,
             is_split=True, ignore_training_inputs=0,
             is_custom_data=False)
    dgm.cuda()

    import os
    os.chdir(home_path + "/" +destination_folder)

    dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2, generate_extra_class=1000 )


if __name__ == "__main__":
    __main__()
