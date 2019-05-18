import numpy as np
def __main__():
    import os

    # os.chdir("..") # To return at the root of the project

    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    dataset_name = "mnist_ssl"
    h_dims = [300, 300]
    h_dims_classifier = [128]
    betas = (0.9, 0.999)
    z_dims = [10]
    a_dims = [10]
    num_elements = 2
    batch_size = 128
    number_of_flows = 10
    input_shape = [1, 28, 28]
    lr = 1e-4
    warmup = 10
    meta_df = None
    unlabelled_meta_df = None
    use_conv = False
    # vae_flavour = "o-sylvester" set in set_vae_flavour.py
    from set_vae_flavour import vae_flavour
    labels_per_class = -1
    early_stopping = 200
    has_unlabelled = False
    load_vae = False
    resume = False # not working... TODO to be corrected, would be useful
    n_epochs = 1000
    auxiliary = True
    classif_weight = 1.0
    repetitions = 10
    clip_grad = 1e-4
    mc = 1
    iw = 1
    l1 = 0
    l2 = 0

    num_classes = 10

    dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, a_dim=a_dims[0],
                                       num_elements=num_elements, use_conv=use_conv)

    dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    if meta_df is not None:
        dgm.import_dataframe(meta_df, batch_size, labelled=True)
        if has_unlabelled:
            dgm.import_dataframe(unlabelled_meta_df, batch_size, labelled=False)
        else:
            dgm.import_dataframe(meta_df, batch_size, labelled=False)
    dgm.load_example_dataset(dataset="mnist", batch_size=batch_size, labels_per_class=labels_per_class,
                             extra_class=True)
    dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour,
                              model_name="mnist_vae", z_dim=z_dims[-1])
    dgm.cuda()

    if use_conv:
        dgm.set_conv_adgm_layers(is_hebb_layers=False, input_shape=input_shape)
    else:
        dgm.set_adgm_layers(h_dims=h_dims_classifier, input_shape=input_shape, num_classes=num_classes)
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
    log_likehihoods = dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2, t_max=classif_weight,
            verbose=1, generate_extra_class=1000, clip_grad=clip_grad, times=repetitions)
    mean_ll = np.mean(log_likehihoods)
    print("Mean:", mean_ll)
if __name__ == "__main__":
    __main__()
