
def __main__():

    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    from utils.files_destinations import home_path, destination_folder, data_folder, results_folder

    dataset_name = "mnist_ssl_conv"
    n_epochs = 1000
    lr = 1e-4
    h_dims = [64, 32]
    h_dims_classifier = [128]
    betas = (0.9, 0.999)
    z_dims = [20]
    a_dims = [20]
    num_elements = 2
    input_shape = [1, 28, 28]
    vae_flavour = "o-sylvester"
    number_of_flows = 4
    batch_size = 64
    warmup = 3
    load_vae = False
    resume = False
    labels_per_class = -1
    early_stopping = 100
    use_conv = True
    auxiliary = True
    mc = 1
    iw = 1
    l1 = 0.
    l2 = 0.

    dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, a_dim=a_dims[0],
                                       num_elements=num_elements, use_conv=use_conv)

    dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    dgm.load_example_dataset(dataset="mnist", batch_size=batch_size, labels_per_class=labels_per_class)
    dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour)
    dgm.cuda()

    dgm.set_conv_adgm_layers(is_hebb_layers=False, input_shape=input_shape)

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
    dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2)

if __name__ == "__main__":
    __main__()
