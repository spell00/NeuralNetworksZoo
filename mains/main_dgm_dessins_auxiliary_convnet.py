from numpy import genfromtxt
from torchvision import transforms, datasets
import torch
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

def __main__():
    from data_preparation.GeoParser import GeoParser
    from dimension_reduction.ordination import ordination2d
    from sklearn.decomposition import PCA
    from IPython.display import Image
    import pandas as pd
    import numpy as np

    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    from utils.utils import dict_of_int_highest_elements, plot_evaluation

    # files_destinations
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    meta_destination_folder = "pandas_meta_df"

    plots_folder_path = "/".join([home_path, destination_folder, results_folder, "plots/"])

    #dataset_name = "gse33000_and_GSE24335_GSE44768_GSE44771_GSE44770"
    dataset_name = "dessins"
    activation = "relu"
    #nrep = 3
    betas=(0.9, 0.999)
    vae_flavour = "o-sylvester"
    early_stopping = 200
    labels_per_class = 10000
    n_epochs = 1000
    warmup = 100
    gt_input = 10000

    # if ladder is yes builds a ladder vae. Do not combine with auxiliary (yet; might be possible and relatively
    # not too hard to implement, but might be overkill. Might be interesting too)
    translate = "n"

    # Types of deep generative model

    # Convolution neural network (convolutional VAE and convolutional classifier)
    use_conv_ae = False #Not applicable if not sequence (images, videos, sentences, DNA...)
    use_convnet = True
    # Ladder VAE (L-VAE)
    ladder = False

    # Auxiliary Variational Auto-Encoder (A-VAE)
    auxiliary = True

    # Load pre-computed vae (unsupervised learning)
    load_vae = False

    lr = 1e-4
    l1 = 0.
    l2 = 0.
    batch_size = 128
    mc = 1 # seems to be a problem when mc > 1 for display only, results seem good
    iw = 1 # seems to be a problem when iw > 1 for display only, results seem good

    # Neurons layers
    a_dim = 50
    h_dims_classifier = [256]
    h_dims = [256, 128]
    z_dims = [50]

    # number of flows
    number_of_flows = 5
    num_elements = 2


    # Files destinations
    load_from_disk = True
    load_merge = False
    home_path = "/home/simon/"
    destination_folder = "annleukemia"
    data_folder = "data"
    results_folder = "results"
    meta_destination_folder = "pandas_meta_df"
    plots_folder_path = "/".join([home_path, destination_folder,
                                  results_folder, "plots/"])

    dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows,a_dim=a_dim,
                                       num_elements=num_elements, is_hebb_layers=True,
                                       gt_input=gt_input)

    dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers",
                    is_unlabelled=True)

    dgm.load_local_dataset(root_train="/home/simon/annleukemia/data/kaggle_dessins/train",
                           root_valid="/home/simon/annleukemia/data/kaggle_dessins/valid",
                           root_test="/home/simon/annleukemia/data/kaggle_dessins/test", n_classes=31,
                           batch_size=batch_size, labels_per_class=labels_per_class,
                           extra_class=True, unlabelled_train_ds=True, normalize=False, mu=0.5, var=0.5)

    is_example = False
    # GET ordination from this!
    train = np.vstack([x[0].data.numpy() for x in dgm.x_train])
    # unlabelled_train = np.vstack([x[0].data.numpy() for x in dgm.unlabelled_x_train])

    targets = np.vstack([x[1].data.numpy() for x in dgm.x_train])
    labels = [x.tolist().index(1) for x in targets]

    dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour)
    dgm.set_data(labels_per_class=labels_per_class, is_example=True, extra_class=True)

    planes_classifier = [1, 16, 32, 64, 128, 256, 512]
    classifier_kernels = [3, 3, 3, 3, 3, 3, 3]
    classifier_pooling_layers = [True, True, True, True, True, True, False, False]

    dgm.set_adgm_layers(h_dims=h_dims_classifier, input_shape=[1, 100, 100], use_conv_classifier=use_convnet,
                        planes_classifier=planes_classifier,
                        classifier_kernels=classifier_kernels,
                        classifier_pooling_layers=classifier_pooling_layers)

    #dgm.set_dgm_layers()

    # import the M1 in the M1+M2 model (Kingma et al, 2014). Not sure if it still works...
    if load_vae:
        print("Importing the model: ", dgm.model_file_name)
        if use_conv_ae:
            dgm.import_cvae()
        else:
            dgm.load_model()
        # dgm.set_dgm_layers_pretrained()
    dgm.cuda()
    # dgm.vae.generate_random(False, batch_size, z1_size, [1, 28, 28])
    dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2, verbose=1,
            show_progress=10, show_pca_train=10, show_lda_train=10, show_pca_generated=10, clip_grad=1e-5,
            is_input_pruning=False, start_pruning=10000, show_lda_generated=10, warmup_n=-1, alpha_rate=1000)


if __name__ == "__main__":
    __main__()
