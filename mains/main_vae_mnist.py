import numpy as np
from models.generative.autoencoders.vae.vae import VariationalAutoencoder
from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE

# files_destinations
home_path = "/home/simon/"
destination_folder = "annleukemia"
data_folder = "data"
results_folder = "results"
meta_destination_folder = "pandas_meta_df"

plots_folder_path = "/".join([home_path, destination_folder, results_folder, "plots/"])

#dataset_name = "gse33000_and_GSE24335_GSE44768_GSE44771_GSE44770"
dataset_name = "mnist_vae"
activation = "relu"
#nrep = 3
betas=(0.9, 0.999)
vae_flavour = "o-sylvester"
early_stopping = 200
n_epochs = 1000
warmup = 10
gt_input = 0

# if ladder is yes builds a ladder vae. Do not combine with auxiliary (yet; might be possible and relatively
# not too hard to implement, but might be overkill. Might be interesting too)
translate = "n"

# Convolution neural network (convolutional VAE and convolutional classifier)
use_conv = False #Not applicable if not sequence (images, videos, sentences, DNA...)

# Ladder VAE (L-VAE)
ladder = False
# Load pre-computed vae (unsupervised learning)
load_vae = False

lr = 3e-4
l1 = 0.
l2 = 0.
batch_size = 128
mc = 1 # seems to be a problem when mc > 1 for display only, results seem good
iw = 1 # seems to be a problem when iw > 1 for display only, results seem good

# Neurons layers
h_dims = [300, 300]
z_dims = [2]

# number of flows
number_of_flows = 10
num_elements = 2

is_example = True


def __main__():
    if vae_flavour in ["o-sylvester", "h-sylvester", "t-sylvester"]:
        print("vae_flavour", vae_flavour)
        vae = SylvesterVAE(vae_flavour, z_dims=z_dims, h_dims=h_dims, n_flows=number_of_flows,
                           num_elements=num_elements, auxiliary=False, a_dim=0)
    else:
        print("vae_flavour", vae_flavour)
        vae = VariationalAutoencoder(vae_flavour, z_dims=z_dims, h_dims=h_dims, n_flows=number_of_flows, auxiliary=False,
                                     a_dim=0)

    vae.load_example_dataset(dataset="mnist", batch_size=batch_size, labels_per_class=0, extra_class=False,
                             unlabelled_train_ds=None, normalize=True, mu=0.1307, var=0.3081, unlabelled_samples=False)

    train = np.vstack([x[0].data.numpy() for x in vae.x_train])
    # unlabelled_train = np.vstack([x[0].data.numpy() for x in dgm.unlabelled_x_train])

    targets = np.vstack([x[1].data.numpy() for x in vae.x_train])
    labels = [x.tolist().index(1) for x in targets]
    vae.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    vae.define_configurations(vae_flavour, early_stopping=1000, warmup=warmup, ladder=ladder, z_dim=z_dims[-1],
                              auxiliary=False, supervised="no", l1=l1, l2=l2, model_name="mnist_vae")

    vae.set_data(is_example=is_example, labels_per_class=0)
    if ladder:
        print("Setting ladder layers")
        vae.set_lvae_layers()
    else:
        vae.set_vae_layers(sample_layer="")

    if load_vae:
        vae.load_model()

    vae.run(epochs=n_epochs, gen_rate=1, clip_grad=0., lambda1=l1, lambda2=l2)


if __name__ == "__main__":
    __main__()
