# idea: use the likelihood found in the VAE to "weight" the classifier resulting of the algorithm. Could be used if
# Multiple classifiers are obtained and we want to bag them; instead of an empirical mean, use it for weighted mean

def __main__():

    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    from models.semi_supervised.deep_generative_models.models.ladder_dgm import LadderDeepGenerativeModel
    from models.semi_supervised.deep_generative_models.models.dgm import DeepGenerativeModel
    from utils.files_destinations import home_path, destination_folder, data_folder, results_folder

    h_dims = [64, 32]
    h_dims_classifier = [128]
    betas = (0.9, 0.999)
    z_dims = [20]
    a_dims = [20]
    num_elements = 2
    img_shape = [1, 28, 28]
    meta_df = None
    unlabelled_meta_df = None

    labels_per_class = -1
    early_stopping = 100

    if auxiliary:
        dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows,a_dim=a_dims[0],
                                           num_elements=num_elements)

        dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                        destination_folder=destination_folder, dataset_name=dataset_name, lr=initial_lr,
                        meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    elif ladder:
        dgm = LadderDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, auxiliary=False)

        dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                        destination_folder=destination_folder, dataset_name=dataset_name, lr=initial_lr,
                        meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")
    else:
        print(vae_flavour)
        dgm = DeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, a_dim=None, auxiliary=False,
                                  num_elements=num_elements)

        dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                        destination_folder=destination_folder, dataset_name=dataset_name, lr=initial_lr,
                        meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")
    if meta_df is not None:
        dgm.import_dataframe(meta_df, batch_size, labelled=True)
        if has_unlabelled:
            dgm.import_dataframe(unlabelled_meta_df, batch_size, labelled=False)
        else:
            dgm.import_dataframe(meta_df, batch_size, labelled=False)
    dgm.load_example_dataset(dataset="mnist", batch_size=batch_size, labels_per_class=labels_per_class)
    dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour)
    dgm.cuda()

    if auxiliary:
        if use_conv:
            dgm.set_conv_adgm_layers(is_hebb_layers=False, input_shape=img_shape)
        else:
            dgm.set_adgm_layers(h_dims=h_dims_classifier)
    elif ladder:
        dgm.set_ldgm_layers()
    else:
        if use_conv:
            dgm.set_conv_dgm_layers(input_shape=img_shape)
        else:
            dgm.set_dgm_layers()

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

    dgm.cuda()
    # dgm.vae.generate_random(False, batch_size, z1_size, [1, 28, 28])
    dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2 )

if __name__ == "__main__":
    from utils.parameters import *
    from utils.list_parameters import *
    __main__()
