import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--bad_geo_ids', default="") # Results were not good
parser.add_argument('--geo_ids', default="GSE33000")
parser.add_argument('--unlabelled_geo_ids', default="GSE33000 GSE24335 GSE44768 GSE44771 GSE44770") # GSE12649 GSE84422 GSE45480 GSE11863 GSE23314
parser.add_argument("--lr", help="Learning Rate", default=1e-3)
parser.add_argument("--l1", help="l1 regularization", default=0.01)
parser.add_argument("--l2", help="l2 regularization", default=0.000001)
parser.add_argument("--z_dim_last", default=20)
parser.add_argument("--a_dim", default=20)
parser.add_argument("--n_combinations", default=20)
parser.add_argument("--mc", default=1)
parser.add_argument("--iw", default=1)
parser.add_argument("--number_of_flows", default=4)
parser.add_argument("--num_elements", default=8) # For Sylvester flows only
parser.add_argument("--activation", help="Activation Function", default="relu") # TODO Does nothing yet, to implement
parser.add_argument("--nrep", help="Number of repetitions", default=2) # TODO Does nothing yet, to implement
parser.add_argument("--init", help="Which initialization to use? (default: he)", default='he_uniform')
parser.add_argument("--n_epochs", help="How many epochs?", default=100)
parser.add_argument("--batch_size", type=int, help="What is the batch size?", default=8)
parser.add_argument("--hidden_size", help="How many neurons per layer", default=128)
parser.add_argument("--has_nvidia", help="Do you have a NVIDIA gpu?", default='y')
parser.add_argument("--silent", help="silent?", default="y")
parser.add_argument("--load_from_disk", help="Load from disk?", default='y')
parser.add_argument("--load_merge", help="Load from disk?", default='y')
parser.add_argument("--translate", help="The ids need to be translated to something else?", default='n')
parser.add_argument("--nets", default="dgm")
parser.add_argument("--ords", help="Which ordinations to use? [pca/tsne/both]", default="none")
parser.add_argument("--vae_flavour", help="Which flavour of variational autoencoder?", default='vanilla')
parser.add_argument("--all_automatic", default='n')
parser.add_argument("--evaluate_individually", default='n')
parser.add_argument("--example", default='')
parser.add_argument("--ask", default='n')
parser.add_argument("--load_vae", default='n')
parser.add_argument("--vae_path", default='n')
parser.add_argument("--warmup", default='100')
parser.add_argument("--ladder", default='n')
parser.add_argument("--auxiliary", default='y')
parser.add_argument("--use_conv", default='n')
parser.add_argument("--resume_training", default='n')

args = parser.parse_args()
ladder = args.ladder
resume = args.resume_training
use_conv = args.use_conv
auxiliary = args.auxiliary
vae_path = args.vae_path
load_merge = args.load_merge
load_vae = args.load_vae
warmup = int(args.warmup)
ask = args.ask
mc = int(args.mc)
iw = int(args.iw)
a_dim = int(args.a_dim)
number_of_flows = int(args.number_of_flows)
n_combinations = int(args.n_combinations)
ords = args.ords
vae_flavour = args.vae_flavour
example = args.example
evaluate_individually = args.evaluate_individually
translate = args.translate
bad_geo_ids = args.bad_geo_ids.split(" ")
bad_geo_ids = [x for x in bad_geo_ids if x is not "" or x is not " " or x is not None]
geo_ids = args.geo_ids.split(" ")
geo_ids = [x for x in geo_ids if x is not "" or x is not " " or x is not None]
unlabelled_geo_ids = args.unlabelled_geo_ids.split(" ")
unlabelled_geo_ids = [x for x in unlabelled_geo_ids if x is not "" or x is not " " or x is not None]
nvidia = args.has_nvidia
initial_lr = args.lr
init = args.init
n_epochs = int(args.n_epochs)
batch_size = args.batch_size
hidden_size = args.hidden_size
nrep = args.nrep
load_from_disk = args.load_from_disk
activation = args.activation
l1 = args.l1
l2 = args.l2
silent = args.silent
nets = args.nets.split(" ")
if auxiliary == "y":
    auxiliary = True
else:
    auxiliary = False
if ladder == "y":
    ladder = True
else:
    ladder = False
print("LADDER", ladder)
if load_vae is "y":
    load_vae = True
else:
    load_vae = False
if load_merge is "y":
    load_merge = True
else:
    load_merge = False
if use_conv is "y":
    use_conv = True
else:
    use_conv = False
if evaluate_individually is "y":
    evaluate_individually = True
else:
    evaluate_individually = False
if args.all_automatic is 'y':
    all_automatic = True
else:
    all_automatic = False
if silent is 'y':  # TODO it should be the other way...
    silent = True
else:
    silent = False
if resume is 'y':  # TODO it should be the other way...
    resume = True
else:
    resume = False
if load_from_disk is 'y':
    load_from_disk = True
else:
    load_from_disk = False

# Install the plaidml backend
if nvidia != 'y':
    print('Using plaidml')
    import plaidml.keras

    plaidml.keras.install_backend()
else:
    torch.backends.cudnn.benchmark = True
if example == "cifar10":
    n_channels = 3
if example == "mnist":
    n_channels = 1

dataset_name = "_".join(geo_ids) + "-" + "_".join(unlabelled_geo_ids)

dataset_name = "toto"
