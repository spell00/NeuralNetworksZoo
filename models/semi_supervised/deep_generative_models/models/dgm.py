import torch.nn as nn
from torch.nn import init
from scipy.stats import norm

import torch
import torchvision as tv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.semi_supervised.utils.utils import onehot_array
from models.generative.autoencoders.vae.vae import Encoder, Decoder, ConvDecoder, ConvEncoder
from models.discriminative.artificial_neural_networks.MultiLayerPerceptron import MLP
from models.discriminative.artificial_neural_networks.ConvNet import ConvNet
from models.utils.utils import create_missing_folders
import pylab
from models.utils import ladder
import pandas as pd
from itertools import cycle
from torch.autograd import Variable
from dimension_reduction.ordination import ordination2d
#from parameters import vae_flavour, ladder
import numpy as np
import torch.backends.cudnn as cudnn
from models.utils.visual_evaluation import plot_histogram
from scipy.special import logsumexp

if torch.cuda.is_available():
    cudnn.enabled = True
    device = torch.device('cuda:0')
else:
    cudnn.enabled = False
    device = torch.device('cpu')

from set_vae_flavour import vae_flavour
if ladder:
    from models.generative.autoencoders.vae.ladder_vae import LadderVariationalAutoencoder as VAE
else:
    if vae_flavour == "o-sylvester":
        from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE as VAE
    elif vae_flavour == "h-sylvester":
        from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE as VAE
    elif vae_flavour == "t-sylvester":
        from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE as VAE
    else:
        from models.generative.autoencoders.vae.vae import VariationalAutoencoder as VAE

def safe_log(z):
    import torch
    return torch.log(z + 1e-7)


def rename_model(model_name, warmup, z1_size, l1, l2):
    model_name = model_name
    if model_name == 'vae_HF':
        number_combination = 0
    elif model_name == 'vae_ccLinIAF':
        number_of_flows = 1

    if model_name == 'vae_HF':
        model_name = model_name + '(T_' + str(number_of_flows) + ')'
    elif model_name == 'vae_ccLinIAF':
        model_name = model_name + '(K_' + str(number_combination) + ')'

    model_name = model_name + '_wu(' + str(warmup) + ')' + '_z' + str(z1_size) + "_l1" \
                 + str(l1) + "_l2" + str(l2)

    return model_name


def plot_performance(loss_total, loss_labelled, loss_unlabelled, accuracy, labels, results_path,
                     filename="NoName", verbose=0):
    fig2, ax21 = plt.subplots()
    try:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:' + str(len(labels["valid"])))  # plotting t, a separately
        ax21.plot(loss_labelled["train"], 'b-.', label='Train labelled loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_labelled["valid"], 'g-.', label='Valid labelled loss:' + str(len(labels["valid"])))  # plotting t, a separately
        ax21.plot(loss_unlabelled["train"], 'b.', label='Train unlabelled loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_unlabelled["valid"], 'g.', label='Valid unlabelled loss:' + str(len(labels["valid"])))  # plotting t, a separately
        #ax21.plot(values["valid"], 'r-', label='Test:' + str(len(labels["valid"])))  # plotting t, a separately
    except:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:')  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:')  # plotting t, a separately
        ax21.plot(loss_labelled["train"], 'b-.', label='Train labelled loss:')  # plotting t, a separately
        ax21.plot(loss_labelled["valid"], 'g-.', label='Valid labelled loss:')  # plotting t, a separately
        ax21.plot(loss_unlabelled["train"], 'b.', label='Train unlabelled loss:')  # plotting t, a separately
        ax21.plot(loss_unlabelled["valid"], 'g.', label='Valid unlabelled loss:')  # plotting t, a separately

    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Loss')
    handles, labels = ax21.get_legend_handles_labels()
    ax21.legend(handles, labels)
    ax22 = ax21.twinx()

    #colors = ["b", "g", "r", "c", "m", "y", "k"]
    # if n_list is not None:
    #    for i, n in enumerate(n_list):
    #        ax22.plot(n_list[i], '--', label="Hidden Layer " + str(i))  # plotting t, a separately
    ax22.set_ylabel('Accuracy')
    ax22.plot(accuracy["train"], 'b--', label='Train')  # plotting t, a separately
    ax22.plot(accuracy["valid"], 'g--', label='Valid')  # plotting t, a separately
    handles, labels = ax22.get_legend_handles_labels()
    ax22.legend(handles, labels)

    fig2.tight_layout()
    # pylab.show()
    if verbose > 0:
        print("Performance at ", results_path)
    create_missing_folders(results_path + "/plots/")
    pylab.savefig(results_path + "/plots/" + filename)
    plt.show()
    plt.close()



class DeepGenerativeModel(VAE):
    def __init__(self, flow_type, z_dims, h_dims, n_flows, a_dim, auxiliary, supervised, is_pruning,
                 labels_per_class, num_elements=None, n_h=4, dropout=0.5, gt_input=-100):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        self.pretrained = False
        self.gt_input = gt_input
        self.is_pruning = is_pruning
        self.a_dim = a_dim
        self.flow_type = flow_type
        self.dropout = dropout
        self.auxiliary = auxiliary
        self.n_h = n_h
        super(DeepGenerativeModel, self).__init__(flavour=flow_type, z_dims=z_dims, h_dims=h_dims, auxiliary=auxiliary,
                                                  n_flows=n_flows, num_elements=num_elements, a_dim=a_dim,
                                                  is_pruning=is_pruning)
        self.z_dim, self.h_dims = z_dims[-1], h_dims
        self.n_flows = n_flows
        self.num_elements = num_elements
        self.labels_per_class = labels_per_class

        self.train_total_loss_history = []
        self.train_labelled_loss_history = []
        self.train_unlabelled_loss_history = []
        self.train_accuracy_history = []
        self.train_kld_history = []
        self.valid_total_loss_history = []
        self.valid_labelled_loss_history = []
        self.valid_unlabelled_loss_history = []
        self.valid_accuracy_history = []
        self.valid_kld_history = []
        self.hebb_input_values_history = []
        self.epoch = 0
        self.indices_names = None
        self.use_conv = False
        self.supervised = supervised

    def set_dgm_layers(self, input_shape, num_classes, is_clamp=False):
        import numpy as np
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_size = np.prod(input_shape)
        self.set_vae_layers()
        self.encoder = Encoder(input_size=self.input_size, h_dim=self.h_dims, z_dim=self.z_dim,
                               num_classes=self.num_classes, y_dim=self.num_classes)
        self.decoder = Decoder(self.z_dim, list(reversed(self.h_dims)), self.input_size, num_classes=self.num_classes)

        hs = [self.h_dims[0] for _ in range(self.n_h)]
        if self.indices_names is None:
            self.indices_names = list(range(self.input_size))
        # The extra_class is previously added; this would put a second extra-class
        self.classifier = MLP(self.input_size, self.input_shape, self.indices_names, hs, self.num_classes,
                              dropout=self.dropout, is_pruning=self.is_pruning, is_clamp=is_clamp, gt_input=self.gt_input,
                              extra_class=False, destination_folder=self.home_path+"/"+self.destination_folder)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_conv_dgm_layers(self, hs_ae, hs_class, z_dim, planes_ae, kernels_ae, padding_ae, pooling_layers_ae,
                            planes_c, kernels_c, pooling_layers_c, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        self.set_cvae_layers(hs_ae, z_dim, planes_ae, kernels_ae, padding_ae, pooling_layers_ae)
        self.encoder = ConvEncoder(h_dim=hs_class, z_dim=z_dim, planes=planes_ae, kernels=kernels_ae,
                                   padding=padding_ae, pooling_layers=pooling_layers_ae)
        self.decoder = ConvDecoder(z_dim=z_dim, num_classes=self.num_classes, h_dim=list(reversed(hs_ae)), input_shape=self.input_shape,
                                   planes=list(reversed(planes_ae)), kernels=list(reversed(kernels_ae)),
                                   padding=list(reversed(padding_ae)), unpooling_layers=pooling_layers_ae)

        self.classifier = ConvNet(self.input_size, hs_class, self.num_classes, planes=planes_c, kernels=kernels_c,
                                  pooling_layers=pooling_layers_c, a_dim=self.a_dim, extra_class=False,
                                  indices_names=list(range(self.input_size)))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_dgm_layers_pretrained(self):
        classes_tensor1 = init.xavier_uniform_(torch.zeros([self.encoder.hidden[0].weight.size(0), self.num_classes]))
        classes_tensor2 = init.xavier_uniform_(torch.zeros([self.decoder.hidden[0].weight.size(0), self.num_classes]))
        self.encoder.hidden[0].weight = nn.Parameter(torch.cat((self.encoder.hidden[0].weight, classes_tensor1.cuda()), dim=1))
        self.decoder.hidden[0].weight = nn.Parameter(torch.cat((self.decoder.hidden[0].weight, classes_tensor2), dim=1))
        self.cuda()

    def classify(self, x, valid_bool, input_pruning=True, is_balanced_relu=False, start_pruning=3):
        print("x", x.shape)
        print("valid_bool", valid_bool.shape)
        logits = self.classifier(x, input_pruning=True, valid_bool=valid_bool, is_balanced_relu=is_balanced_relu)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()

        x = self.decoder(z, y=y)
        return x

    def generate_random(self, epoch=0, verbose=0, show_pca=1, show_lda=1, n=40, drop_na=False, keep_images=True,
                        only_na=False):
        df = None
        colnames = None

        if not only_na:
            with torch.no_grad():
                images_grid = None
                images = None
                #hparams_string = "/".join(["num_elements" + str(self.num_elements), "n_flows" + str(self.n_flows),
                #                           "z_dim" + str(self.z_dim_last), "a_dim" + str(self.a_dim), "lr" + str(self.lr),
                #                           "ladder" + str(self.ladder), self.flavour, 'generate_extra:'+str(self.generate_extra_class)])
                images_path = self.hparams_string + "/random/"
                create_missing_folders(images_path)
                if verbose > 0:
                    print("GENERATING IMAGES AT", images_path)
                # self.eval()
                rand_z = Variable(torch.randn(n * self.num_classes, self.z_dim))

                if not only_na:
                    y = torch.cat(
                        [torch.Tensor(onehot_array(n * [i], self.num_classes)) for i in range(self.num_classes)])
                else:
                    y = torch.Tensor(onehot_array(rand_z.shape[0] * [self.num_classes - 1], self.num_classes))

                rand_z, y = rand_z.cuda(), y.cuda()
                x_mu = self.sample(rand_z, y)

                self.plot_z_stats(rand_z.detach().cpu().numpy(), generate="generated")

                if len(self.input_shape) > 1:
                    images = x_mu.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
                    images_grid = tv.utils.make_grid(images, 20)
                if keep_images and images is not None:
                    tv.utils.save_image(images_grid,
                                        images_path + "/" + str(epoch) + "only_na:" + str(only_na) + "_generated.png")
                colnames = [list(self.labels_set)[one_hot.cpu().numpy().tolist().index(1)] for one_hot in y]
                df = pd.DataFrame(x_mu.transpose(1, 0).detach().cpu().numpy(), columns=colnames)
                if not only_na:
                    if drop_na:
                        try:
                            df = df.drop(["N/A"], axis=1)
                        except:
                            pass
                    if show_pca != 0 and epoch % show_pca == 0 and epoch != 0:
                        try:
                            ordination2d(df, "pca", epoch=self.epoch, images_folder_path=images_path,
                                         dataset_name=self.dataset_name, a=0.5,
                                         verbose=0, info="generated", targets=colnames)
                        except:
                            if verbose > 0:
                                print("No pca.")
                    if show_lda != 0 and epoch % show_lda == 0 and epoch != 0:
                        try:
                            ordination2d(df, "lda", epoch=self.epoch, images_folder_path=images_path,
                                         dataset_name=self.dataset_name, a=0.5,
                                         verbose=0, info="generated", targets=colnames)
                        except:
                            if verbose > 0:
                                print("NO lda")
        else:
            images_grid = None
            images = None
            this_path = self.hparams_string + "/random/"
            create_missing_folders(this_path)
            if verbose > 0:
                print("GENERATING N/A IMAGES AT", this_path)
            # self.eval()
            rand_z = Variable(torch.randn(n * self.num_classes, self.z_dim))

            if not only_na:
                y = torch.cat([torch.Tensor(onehot_array(n * [i], self.num_classes)) for i in range(self.num_classes)])
            else:
                y = torch.Tensor(onehot_array(rand_z.shape[0] * [self.num_classes - 1], self.num_classes))

            rand_z, y = rand_z.cuda(), y.cuda()
            x_mu = self.sample(rand_z, y)

            self.plot_z_stats(rand_z.detach().cpu().numpy(), generate="generated")

            if len(self.input_shape) > 1:
                images = x_mu.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
                images_grid = tv.utils.make_grid(images, 20)
            if keep_images and images is not None:
                tv.utils.save_image(images_grid,
                                    this_path + "/" + str(epoch) + "only_na:" + str(only_na) + "_generated.png")
            colnames = [list(self.labels_set)[one_hot.cpu().numpy().tolist().index(1)] for one_hot in y]
            df = pd.DataFrame(x_mu.transpose(1, 0).detach().cpu().numpy(), columns=colnames)
            if not only_na:
                if drop_na:
                    try:
                        df = df.drop(["N/A"], axis=1)
                    except:
                        pass
                if show_pca != 0 and epoch % show_pca == 0 and epoch != 0:
                    try:
                        ordination2d(df, "pca", epoch=self.epoch, images_folder_path=this_path,
                                     dataset_name=self.dataset_name, a=0.5,
                                     verbose=0, info="generated")
                    except:
                        if verbose > 0:
                            print("No pca.")
                if show_lda != 0 and epoch % show_lda == 0 and epoch != 0:
                    try:
                        ordination2d(df, "lda", epoch=self.epoch, images_folder_path=this_path,
                                     dataset_name=self.dataset_name, a=0.5,
                                     verbose=0, info="generated")
                    except:
                        if verbose > 0:
                            print("NO lda")
        del df, colnames, images_grid, x_mu, rand_z

        return images, y

    def plot_z_stats(self, z, generate="generated"):
        fig, ax = plt.subplots()  # create figure and axis
        plt.boxplot(z)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.tight_layout()
        fig.tight_layout()
        path = "/".join([self.results_path, "plots/vae_z_stats"]) + "/"
        fig.savefig(path + self.flavour + "_" + str(self.epoch) + '_lr' + str(self.lr) + '_bs' + str(self.batch_size)
                    + "_" + generate + ".png")
        print("SAVED: z_stats at: ", path)
        plt.close(fig)


    def generate_uniform_gaussian_percentiles(self, epoch=0, verbose=0, show_pca=0, show_lda=0, n=20, drop_na=False):
        zs_grid = torch.stack([torch.Tensor(np.vstack([np.linspace(norm.ppf(0.05), norm.ppf(0.95), n**2)
                                                       for _ in range(self.z_dim_last)]).T)
                               for _ in range(self.num_classes)])

        # I get much better results squeezing values with tanh

        this_path = self.hparams_string + "/gaussian_percentiles/"
        if verbose > 0:
            print("GENERATING SS DGM IMAGES AT", this_path)

        y = torch.stack([torch.Tensor(onehot_array(n**2*[i], self.num_classes)) for i in range(n)])
        x_mu = [self.sample(torch.Tensor(zs_grid[i]).cuda(), y[i]) for i in range(self.num_classes)]

        # self.plot_z_stats(rand_z.detach().cpu().numpy(), generate="generated")
        labels_set_ints = list(range(len(self.labels_set)))
        if len(self.input_shape) > 1:
            images = torch.stack([x_mu[i].view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
                      for i in range(len(x_mu))])
            images = images.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            images_grid = tv.utils.make_grid(images, n)
            create_missing_folders(this_path)
            tv.utils.save_image(images_grid, this_path + "/" + str(epoch) + "gaussian_percentiles_generated.png")

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def run(self, n_epochs, auxiliary, mc=1, iw=1, lambda1=0., lambda2=0., verbose=1, show_progress=0,
            show_pca_train=1, show_lda_train=1, show_pca_valid=1, show_pca_generated=1, clip_grad=0.00001, warmup_n=-1,
            is_input_pruning=False, start_pruning=-1, schedule=True, show_lda_generated=1, is_balanced_relu=False,
            limit_examples=1000, keep_history=True, decay=1e-8, alpha_rate=0.1, t_max=1, generate_extra_class=100,
            times=10, kill_neurites_round=1):
        import os
        from models.semi_supervised.deep_generative_models.inference import SVI, DeterministicWarmup,\
            ImportanceWeightedSampler
        try:
            alpha = alpha_rate * len(self.train_loader_unlabelled) / len(self.train_loader)
        except:
            self.train_loader_unlabelled = self.train_loader
            alpha = alpha_rate * len(self.train_loader_unlabelled) / len(self.train_loader)
        self.generate_extra_class = generate_extra_class
        x_na = torch.Tensor([])
        y_na = torch.Tensor([])

        if torch.cuda.is_available():
            self.cuda()

        self.hparams_string = "/".join([os.getcwd() + "/results/dgm/num_elements"+str(self.num_elements),
                                        "n_flows"+str(self.n_flows), "z_dim"+str(self.z_dim_last), "supervised"+self.supervised,
                                        "pretrained:"+str(self.pretrained),
                                        "labels_per_class"+str(self.labels_per_class),
                                        "extra_class"+str(self.extra_class)])
        if self.supervised == "semi":
            self.hparams_string = "/".join([self.hparams_string, "a_dim"+str(self.a_dim)])

        self.hparams_string = "/".join([self.hparams_string, "lr" + str(self.lr), "ladder" + str(self.ladder),
                                        self.flavour])

        generated_na_examples_path = "/".join([self.hparams_string, "/extra_class_examples/"])
        create_missing_folders(generated_na_examples_path)
        self.valid_bool = [1.] * np.prod(self.input_shape)
        if warmup_n == -1:
            print("Warmup on: ", 4*len(self.train_loader_unlabelled)*100)
            beta = DeterministicWarmup(n=4*len(self.train_loader_unlabelled)*100, t_max=t_max)
        elif warmup_n > 0:
            print("Warmup on: ", warmup_n)
            beta = DeterministicWarmup(n=warmup_n, t_max=t_max)
        else:
            beta = 1.


        sampler = ImportanceWeightedSampler(mc, iw)

        elbo = SVI(self, beta=beta, labels_set=self.labels_set, images_path=self.hparams_string, dataset_name=self.dataset_name,
                   auxiliary=auxiliary, batch_size=self.batch_size, likelihood=F.mse_loss, sampler=sampler,
                   ladder=self.ladder)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True, cooldown=0,
                                                               patience=100)

        best_loss = 100000
        early = 0
        best_accuracy = 0

        self.prints()
        self.zs_train = []
        self.zs_train = []
        flag_na = False
        log_likehihoods = []
        for time in range(times):
            print('Time:', time)
            for epoch in range(n_epochs):
                if len(self.zs_train) > 0 and generate_extra_class != -1:
                    x_na, y_na = self.generate_random(n=100, verbose=0, keep_images=False, only_na=True, epoch=self.epoch)
                    path = generated_na_examples_path + "/" + str(self.epoch) + "/"
                    create_missing_folders(path)
                    # x_na = x_na.view(-1, np.prod(self.input_shape))
                    if x_na is not None:
                        for i, x_n in enumerate(x_na):
                            tv.utils.save_image(x_n, path + str(i) + "_.png")
                        x_na = x_na.reshape(x_na.shape[0], -1)
                    else:
                        print("x_na is None...")
                logits_array = []
                self.zs_train = []
                self.as_train = []
                self.zs_train_targets = []
                self.zs_test = []
                self.as_test = []
                self.zs_test_targets = []
                file = open("logs/" + self.__class__.__name__ + ".log", 'a+')
                file_involvment = open("logs/" + self.__class__.__name__ + "_involvment.log", 'a+')
                self.epoch += 1
                total_loss, labelled_loss, unlabelled_loss, accuracy, accuracy_total = (0, 0, 0, 0, 0)

                print("epoch", epoch, file=file)
                if verbose > 0:
                    print("epoch", epoch)
                c = 0
                recs_train = None
                ys_train = None

                # https://docs.python.org/2/library/itertools.html#itertools.cycle
                # cycle make it so if train_loader_unlabelled is not finished, it will "cycle" (repeats) indefinitely

                # TODO add a module that select a defined number (e.g. 10,000 samples) or a ratio of total images
                # TODO (1/10):
                # How to proceed:
                #
                # Rejection sampling.
                #
                #   1st step: All samples have a p(x_i) associated to it.
                #
                #             a) Initially, they are uniformly distributed
                #             b) After some time, ALL the samples (i) have a p(x_i) calculated.
                #
                #   2nd step: Randomly draw a number (let's call it RN for Random Number) in the range [0, max(p(x_i))].
                #             So, RN ~ Uniform(min=0, max=max(p(x_i)))
                #             a) If RN <= p(x_i), sample is accepted.
                #                   Note: When sampling from a uniform distribution, e.g. in the first(s) iteration
                #                   (we sample with equal probability at the beginning, because nothing is known), then
                #                   the samples with higher probability of "relevance" are chosen with higher
                #                   probability, and possible samples with low probability of being relevant, like a
                #                   an image of white noise "polluting" a dataset like MNIST, would be selected less and
                #                   less often as the loss keeps getting lower. Ideally, if an image gets a lower score
                #                   wrongfully, then it should latter be able to make a comeback, so it should not be
                #                   discarded too kickly (not necessarly discarded, but at some point given a
                #                   probability of being used so low that it will almost never never going to be
                #                   picked, reducing it's negative impact on the model)
                #
                #                   e.g. If 100x more images of white noise are added to MNIST with random labels,
                #                        it will pollute the dataset and no good results will come out of it.
                #                        Hypothesis: by using this rejection-sampling strategy, the real MNIST images
                #                        will come out most of the time and the white noise images will almost never be
                #                        accepted.
                #
                #                        Probable practical difficulty:
                #                           1- If there is much more "polluting" images than relevant images (e.g. MNIST
                #                              with 100,000,000x more white noise images than relevant MNIST images,
                #                              the white noise images are the polluting ones and the actual MNIST
                #                              images the relevant images), then it might take a lot of time i) before
                #                              the relevant images are "identified", and ii) when the MNIST images have
                #                              very high probability of being accepted(the ideal would be they would
                #                              almost certainly be accepted), a lot of the images are going to have to
                #                              be rejected before having the satisfactory amount of sample
                #                              for the iteration.
                #                           2- It might be easier to manage in the semi-supervised settings, as long as
                #                              the labeled dataset can be trusted. Then, the rejection sampling could be
                #                              applied only on unlabeled samples and instead of sampling until a
                #                              satisfactory amount of samples have been selected, it could try a certain
                #                              amount of sampling trials; if none are accepted, then no unlabeled
                #                              would be used on that iteration, as no sample is deemed good to improve
                #                              the understanding of the dataset we are confidant the labels are right.
                #
                #      *Exception: I use labeled samples both for labeled and unlabeled datasets. Because it is trusted
                #                  they are relevant, they will always be automatically used in the unlabeled dataset.
                #                  If it happens that none of the unlabeled samples have any relevant information for
                #                  the samples of interest, which are the labeled ones, then the unlabeled dataset will
                #                  always at least contain the same samples than the labeled dataset (we assume they
                #                  have p(x_i) = 1.0. They are not even in the random draw; again, they are
                #                  automatically accepted.
                #
                #


                self.train()
                for (x, y), (u, _) in zip(cycle(self.train_loader), self.train_loader_unlabelled):
                    if not self.use_conv:
                        x = x.view(-1, np.prod(x.shape[1:]))
                    optimizer.zero_grad()
                    c += len(x)
                    progress = 100 * c / len(self.train_loader_unlabelled) / self.batch_size

                    if verbose > 1:
                        print("\rProgress: {:.2f}%".format(progress), end="", flush=True)

                    # Wrap in variables
                    x, y, u = Variable(x), Variable(y), Variable(u)

                    if torch.cuda.is_available():
                        # They need to be on the same device and be synchronized.
                        x, y = x.cuda(device=0), y.cuda(device=0)
                        u = u.cuda(device=0)
                    if self.epoch > generate_extra_class and generate_extra_class != -1:
                        x = torch.cat((x, x_na), 0)
                        y = torch.cat((y, y_na))
                    # else n/a samples are not used
                    L, rec, z_q, a_q = elbo(x.float(), y.float(), self.valid_bool)
                    if type(z_q) == torch.Tensor:
                        z_q = z_q.detach().cpu().numpy()
                        a_q = a_q.detach().cpu().numpy()
                        self.zs_train += [z_q]
                        self.as_train += [a_q]
                        # y = y.detach().cpu().numpy()
                        # y = np.array([x.tolist().index(1) for x in y])

                    if y is not None:
                        self.zs_train_targets += [np.where(r == 1)[0][0] for r in y.detach().cpu().numpy()]
                    U, _, _, _ = elbo(u.float(), y=None, valid_bool=self.valid_bool)

                    # Add auxiliary classification loss q(y|x)
                    logits = self.classify(x, valid_bool=self.valid_bool, input_pruning=is_input_pruning,
                                           start_pruning=start_pruning)
                    logits_array += [logits]
                    try:
                        classification_loss = torch.sum(y.float() * torch.log(logits + 1e-8), dim=1).mean()
                    except:
                        exit()
                    params = torch.cat([x.view(-1) for x in self.parameters()])
                    l1_regularization = lambda1 * torch.norm(params, 1)
                    l2_regularization = lambda2 * torch.norm(params, 2)

                    if np.isnan(L.item()):
                        print("Problem with the LABELED loss function in dgm.py. Setting the loss to 0")
                        L = Variable(torch.Tensor([0.]).to(device), requires_grad=False)

                    if np.isnan(U.item()):
                        print("Problem with the UNLABELED loss function in dgm.py. Setting the loss to 0")
                        U = Variable(torch.Tensor([0.]).to(device), requires_grad=False)

                    if np.isnan(l1_regularization.item()):
                        print("Problem with the l1 value in dgm.py. Setting to 0")
                        l1_regularization = Variable(torch.Tensor([0.]).to(device), requires_grad=False)

                    if np.isnan(l2_regularization.item()):
                        print("Problem with the l2 value in dgm.py. Setting to 0")
                        l2_regularization = Variable(torch.Tensor([0.]).to(device), requires_grad=False)
                    if np.isnan(classification_loss.item()):
                        print("Problem with the CLASSIFICATION loss function in dgm.py. Setting the loss to 0")
                        classification_loss = Variable(torch.Tensor([0.]).to(device), requires_grad=False)


                    J_alpha = L - alpha * classification_loss + U + l1_regularization + l2_regularization

                    J_alpha.backward()

                    # `clip_grad_norm` helps prevent the exploding gradient problem.
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                    else:
                        pass
                    if np.isnan(J_alpha.item()):
                        print("loss is nan")

                    total_loss += J_alpha.item()
                    labelled_loss += L.item()
                    unlabelled_loss += U.item()

                    _, pred_idx = torch.max(logits, 1)
                    _, lab_idx = torch.max(y, 1)
                    accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                    accuracy += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                    if recs_train is None:
                        ys_train = y
                        recs_train = rec
                    elif recs_train.shape[0] < limit_examples:
                        recs_train = torch.cat([recs_train, rec], dim=0)
                        ys_train = torch.cat([ys_train, y], dim=0)
                    optimizer.step()
                    hebb_round = 1
                    del J_alpha, L, U, x, u, rec, l1_regularization, l2_regularization, pred_idx, lab_idx, classification_loss, _
                self.eval()

                #if epoch % kill_neurites_round == 0:
                #    print("Clearing near-dead neurites")
                #    self.kill_dead_neurites(threshold=1e-8, sparse_tensor=False, verbose=True)

                #if epoch % hebb_round == 0 and epoch != 0:
                #    if self.hebb_layers:
                #        fcs, self.valid_bool = self.classifier.hebb_layers.compute_hebb(total_loss, epoch,
                #                                    results_path=self.results_path, fcs=self.classifier.fcs, verbose=3)
                #        alive_inputs = sum(self.valid_bool)
                #        if alive_inputs < len(self.valid_bool):
                #            print("Current input size:", alive_inputs, "/", len(self.valid_bool))
    #
                #        hebb_input_values = self.classifier.hebb_layers.hebb_input_values
                 #       self.classifier.fcs = fcs

                        # The last positions are for the auxiliary network, if using auxiliary deep generative model
                #        involment_df = pd.concat((involment_df, pd.DataFrame(hebb_input_values.cpu().numpy()[:-self.a_dim],
                 #                                                            index=self.indices_names)), axis=1)
                #        involment_df.columns = [str(a) for a in range(involment_df.shape[1])]
                #        last_col = str(int(involment_df.shape[1])-1)
                #        print("epoch", epoch, "last ", last_col, file=file_involvment)
                #        print(involment_df.sort_values(by=[last_col], ascending=False), file=file_involvment)


                #colnames = [list(self.labels_set)[one_hot.tolist().index(1)] for one_hot in y]
                #new_cols = colnames * iw * mc
                #dataframe = pd.DataFrame(recs_train.transpose(1, 0).detach().cpu().numpy(), columns=new_cols)

                #if show_pca_train > 0 and epoch % show_pca_train == 0 and epoch != 0:
                #    ordination2d(dataframe, "pca", epoch=self.epoch, images_folder_path=self.hparams_string,
                #                 dataset_name=self.dataset_name, a=0.5, verbose=0, info="train", show_images=show_pca_train)
                #if show_lda_train > 0 and epoch % show_lda_train == 0 and epoch != 0:
                #    ordination2d(dataframe, "lda", epoch=self.epoch, images_folder_path=self.hparams_string,
                #                 dataset_name=self.dataset_name, a=0.5, verbose=0, info="train", show_images=show_lda_train)
                self.zs_train = np.vstack(self.zs_train)
                self.as_train = np.vstack(self.as_train)
                self.zs_train_targets = np.array(self.zs_train_targets)
                logits_matrix = torch.stack(logits_array).detach().cpu().numpy()
                logits_matrix = logits_matrix.reshape(-1 ,logits_matrix.shape[2])
                cat_targets = np.array(self.zs_train_targets)

                data_frame_train = pd.DataFrame(self.zs_train, index=self.zs_train_targets)
                if self.extra_class:
                    n = len(self.labels_set) - 1
                ordination2d(data_frame_train, "pca", self.hparams_string + "/pca_z/",
                             self.dataset_name, epoch, targets=cat_targets,
                             labels_set=list(self.labels_set)[:n])
                ordination2d(data_frame_train, "lda", self.hparams_string + "/lda_z/",
                             self.dataset_name, epoch, targets=cat_targets,
                             labels_set=list(self.labels_set)[:n])
                data_frame_train2 = pd.DataFrame(np.concatenate((data_frame_train.values, logits_matrix), 1),
                                                 index=self.zs_train_targets)
                ordination2d(data_frame_train2, "lda", self.hparams_string + "/lda_z&preds/",
                             self.dataset_name, epoch, targets=cat_targets, labels_set=list(self.labels_set)[:n])
                ordination2d(data_frame_train2, "pca", self.hparams_string + "/pca_z&preds/",
                             self.dataset_name, epoch, targets=cat_targets, labels_set=list(self.labels_set)[:n])
                data_frame_train3 = pd.DataFrame(np.concatenate((data_frame_train2.values, self.as_train), 1),
                                                 index=self.zs_train_targets)
                ordination2d(data_frame_train3, "lda", self.hparams_string + "/lda&preds&aux/",
                             self.dataset_name, epoch, targets=cat_targets, labels_set=list(self.labels_set)[:n])
                ordination2d(data_frame_train3, "pca", self.hparams_string + "/pca&preds&aux/",
                             self.dataset_name, epoch, targets=cat_targets, labels_set=list(self.labels_set)[:n])




                self.plot_z_stats(self.zs_train, generate="train_data")
                with torch.no_grad():
                    m = len(self.train_loader_unlabelled)

                    if keep_history:
                        self.train_total_loss_history += [(total_loss / m)]
                        self.train_labelled_loss_history += [(labelled_loss / m)]
                        self.train_unlabelled_loss_history += [(unlabelled_loss / m)]
                        self.train_accuracy_history += [(accuracy / m)]
                        self.train_kld_history += [(torch.sum(self.kl_divergence).item())]

                    print("Epoch: {}".format(epoch), sep="\t", file=file)
                    print("[Train]\t\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, "
                          "accuracy: {:.4f}, kld: {:.1f}".format(total_loss / m, labelled_loss / m, unlabelled_loss / m,
                                            accuracy_total / m, torch.mean(self.kl_divergence).item()), sep="\t", file=file)
                    if verbose > 0:
                        print("[Train]\t\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.4f}, kld: {:.1f}"
                              .format(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy_total / m,
                                      torch.mean(self.kl_divergence).item()))

                    total_loss, labelled_loss, unlabelled_loss, accuracy, accuracy_total = (0, 0, 0, 0, 0)
                    recs_valid = None
                    ys_valid = None
                    for x, y in self.valid_loader:
                        x, y = Variable(x), Variable(y)

                        if torch.cuda.is_available():
                            x, y = x.cuda(device=0), y.cuda(device=0)
                        if not self.use_conv:
                            x = x.view(-1, np.prod(x.shape[1:]))

                        L, rec, z_q_test, a_q_test = elbo(x.float(), y, self.valid_bool, valid=True)
                        self.zs_test_targets += [np.where(r == 1)[0][0] for r in y.detach().cpu().numpy()]

                        U, _, _, _ = elbo(x.float(), y=None, valid_bool=self.valid_bool, valid=True)

                        logits = self.classify(x, valid_bool=self.valid_bool, input_pruning=is_input_pruning,
                                               start_pruning=start_pruning)
                        classification_loss = torch.sum(y.float() * torch.log(logits + 1e-8), dim=1).mean()
                        J_alpha = L - alpha * classification_loss + U # l1_regularization + l2_regularization

                        total_loss += J_alpha.item()
                        labelled_loss += L.item()
                        unlabelled_loss += U.item()

                        _, pred_idx = torch.max(logits, 1)
                        _, lab_idx = torch.max(y, 1)
                        accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                        if len(pred_idx.data.shape) > 1:
                            accuracy += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                        else:
                            accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                        if recs_valid is None:
                            ys_valid = y
                            recs_valid = rec
                        elif recs_train.shape[0] < limit_examples:
                            recs_valid = torch.cat([recs_valid, rec], dim=0)
                            ys_valid = torch.cat([ys_valid, y], dim=0)

                        if type(z_q_test) == torch.Tensor:
                            z_q_test = z_q_test.detach().cpu().numpy()
                            a_q_test = a_q_test.detach().cpu().numpy()
                        self.zs_test += [z_q_test]
                        self.as_test += [a_q_test]

                        del J_alpha, L, U, pred_idx, lab_idx

                    m = len(self.valid_loader)
                    print("[Validation]\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.4f} , kld: {:.1f}"
                          .format(total_loss / m,  labelled_loss / m, unlabelled_loss / m, accuracy / m,
                                  torch.mean(self.kl_divergence)), sep="\t", file=file)
                    if verbose > 0:
                        print("[Validation]\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.4f} , kld: {:.1f}"
                              .format(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m,
                                      torch.mean(self.kl_divergence)))
                    #self.plot_z_stats(self.zs_test, generate="train_data")

                    if keep_history:
                        self.valid_total_loss_history += [(total_loss / m)]
                        self.valid_labelled_loss_history += [(labelled_loss / m)]
                        self.valid_unlabelled_loss_history += [(unlabelled_loss / m)]
                        self.valid_accuracy_history += [(accuracy / m)]
                        self.valid_kld_history += [(torch.sum(self.kl_divergence).item())]

                    # early-stopping

                    if (total_loss < best_loss) and epoch > self.warmup:
                        print("BEST LOSS!", total_loss / m)
                        early = 0
                        best_loss = total_loss / m

                        #self.save_model()
                    if (accuracy / m > best_accuracy) and epoch > self.warmup:
                        best_accuracy = float(accuracy / m)
                        print("BEST ACCURACY!", best_accuracy)
                        early = 0

                        #self.save_model()

                    else:
                        early += 1
                        if early > self.early_stopping:
                            print("Early Stopping.")
                            break

                    if epoch < self.warmup:
                        print("Warmup:", 100 * epoch / self.warmup, "%", sep="\t", file=file)
                        early = 0
                    self.zs_test = np.vstack(self.zs_test)
                    self.zs_test_targets = np.array(self.zs_test_targets)
                    # cat_targets = np.array(self.zs_train_targets)

                    #data_frame_test = pd.DataFrame(self.zs_test, index=self.zs_test_targets)

                    if self.zs_train.shape[1] == 2:
                        self.plot_z()

                    _, _ = self.generate_random(epoch, verbose=1, show_pca=show_pca_generated, only_na=False,
                                                show_lda=show_lda_generated)
                    _, _ = self.generate_random(n=100, verbose=0, keep_images=True, only_na=True, epoch=self.epoch)

                    if len(x.shape) == 3:
                        self.display_reconstruction(epoch, x, rec)
                    try:
                        self.generate_uniform_gaussian_percentiles(epoch)
                    except:
                        print("Did not generate uniform gaussian")
                    total_losses_histories = {"train": self.train_total_loss_history, "valid": self.valid_total_loss_history}
                    labelled_losses_histories = {"train": self.train_labelled_loss_history, "valid": self.valid_labelled_loss_history}
                    unlabelled_losses_histories = {"train": self.train_unlabelled_loss_history, "valid": self.valid_unlabelled_loss_history}
                    accuracies_histories = {"train": self.train_accuracy_history, "valid": self.valid_accuracy_history}
                    labels = {"train": self.labels_train, "valid": self.labels_test}
                    if show_progress > 0 and epoch % show_progress == 0 and epoch != 0:
                        plot_performance(loss_total=total_losses_histories,
                                     loss_labelled=labelled_losses_histories,
                                     loss_unlabelled=unlabelled_losses_histories,
                                     accuracy=accuracies_histories,
                                     labels=labels,
                                     results_path=self.hparams_string + "/",
                                     filename=self.dataset_name)
                    if schedule:
                        scheduler.step(total_loss)
                    file.close()
                    file_involvment.close()

                    del total_loss, labelled_loss, unlabelled_loss, accuracy, self.kl_divergence, recs_train, \
                        rec, ys_train, ys_valid, recs_valid, accuracies_histories
            test_data = Variable(self.test_loader.dataset.parent_ds.test_data)
            test_data = test_data.view(-1, np.prod(np.array(self.input_shape)))
            test_target = Variable(self.valid_loader.dataset.parent_ds.test_labels)
            full_data = Variable(self.train_loader.dataset.train_data)

            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

            full_data = full_data.data.cpu().float().cuda()
            test_data = test_data.data.cpu().float().cuda()
            # full_data = torch.bernoulli(full_data)
            # test_data = torch.bernoulli(test_data)

            full_data = Variable(full_data.double(), requires_grad=False)

            likelihood = self.calculate_likelihood(test_data, ys=test_target)
            print("test Likelihood:", likelihood)
            log_likehihoods += [likelihood]
        return log_likehihoods

    def define_configurations(self, flavour, early_stopping=100, warmup=100, ladder=True, z_dim=40, epsilon_std=1.0,
                              model_name="vae", init="glorot", optim_type="adam", l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

        self.flavour = flavour
        self.epsilon_std = epsilon_std
        self.warmup = warmup
        self.early_stopping = early_stopping

        # importing model
        self.model_file_name = rename_model(model_name, warmup, z_dim, self.l1, self.l2)

        if self.has_cuda:
            self.cuda()

    def forward(self, x, y=None):
        # Add label and data and generate latent variable
        if len(x.shape) < len(self.input_shape):
            x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        (z, z_mu, z_log_var), h = self.encoder(x, y=y)
        self.kl_divergence = self._kld(z, (z_mu, z_log_var), i=0, h_last=h)
        # Reconstruct data point from latent data and label
        x_mu = self.decoder(z, y=y)

        return x_mu

    def load_model(self):
        print("LOADING PREVIOUSLY TRAINED VAE and classifier")
        trained_vae = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.state_dict')
        trained_classifier = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + 'classifier.state_dict')
        self.load_state_dict(trained_vae)
        self.classifier.load_state_dict(trained_classifier)
        self.epoch = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
        self.train_total_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_total_loss')
        self.train_labelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_labelled_loss')
        self.train_unlabelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_unlabelled_loss')
        self.train_accuracy_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_accuracy')
        self.train_kld_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kld')
        self.valid_total_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_total_loss')
        self.valid_labelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_labelled_loss')
        self.valid_unlabelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_unlabelled_loss')
        self.valid_accuracy_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_accuracy')
        self.valid_kld_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_kld')

    def save_model(self):
        # SAVING
        print("MODEL (with classifier) SAVED AT LOCATION:", self.model_history_path)
        create_missing_folders(self.model_history_path)
        torch.save(self.state_dict(), self.model_history_path + self.flavour + "_" + self.model_file_name +'.state_dict')
        torch.save(self.classifier.state_dict(), self.model_history_path + self.flavour + "_" + self.model_file_name +'classifier.state_dict')
        torch.save(self.train_total_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_total_loss')
        torch.save(self.train_labelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_labelled_loss')
        torch.save(self.train_unlabelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_unlabelled_loss')
        torch.save(self.train_accuracy_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_accuracy')
        torch.save(self.train_kld_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kld')
        torch.save(self.valid_total_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_total_loss')
        torch.save(self.valid_labelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_labelled_loss')
        torch.save(self.valid_unlabelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_unlabelled_loss')
        torch.save(self.valid_accuracy_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_accuracy')
        torch.save(self.valid_kld_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_kld')
        torch.save(self.epoch, self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
        #torch.save(self.test_log_likelihood, self.model_history_path + self.flavour + '.test_log_likelihood')
        #torch.save(self.test_loss, self.model_history_path + self.flavour + '.test_loss')
        #torch.save(self.test_re, self.model_history_path + self.flavour + '.test_re')
        #torch.save(self.test_kl, self.model_history_path + self.flavour + '.test_kl')

    def calculate_likelihood(self, data, mode='valid', s=5000, display_rate=100, ys=None, As=None):
        # set auxiliary variables for number of training and valid sets
        n_test = data.size(0)

        # init list
        likelihood = []

        mb = 500

        if s <= mb:
            r = 1
        else:
            r = s / mb
            s = mb

        for j in range(n_test):
            n = 100 * (j / (1. * n_test))
            if j % display_rate == 0 and j != 0:
                print("\revaluating likelihood:", j, "/", n_test, -np.mean(likelihood), end="", flush=True)
            # Take x*                    print("\rProgress: {:.2f}%".format(progress), end="", flush=True)
            x_single = data[j].unsqueeze(0).view(self.input_shape[0], self.input_size)
            y_single = [ys[j]]
            y_single = torch.Tensor(onehot_array(y_single, self.num_classes)).cuda()
            a_list = []
            for _ in range(0, int(r)):
                # Repeat it for all training points
                x = x_single.expand(s, x_single.size(1))
                y = y_single.expand(s, y_single.size(1))

                # pass through VAE
                # if self.flavour in ["ccLinIAF", "hf", "vanilla", "normflow"]:
                #(q_a, a_mu, a_log_var), _ = self.aux_encoder(x, input_shape=self.input_shape)
                #(z, z_mu, z_log_var), h = self.encoder(x, input_shape=self.input_shape, y=y, a=q_a)
                # Generative p(x|z,y)
                #x_mean = self.decoder(z, y)

                # Generative p(a|z,y,x)
                #(p_a, p_a_mu, p_a_log_var), h_a = self.aux_decoder(x=x, input_shape=self.input_shape, y=y, a=z)

                loss, rec, kl, z_q, _ = self.calculate_losses(x, ys=y)
                # elif self.flavour in ["o-sylvester", "h-sylvester", "t-sylvester"]:
                #    reconstruction, z_mu, z_var, ldj, z0, zk = self(x, y)  # couper le y en section de 500 comme le reste
                #    loss, rec, kl = calculate_losses(reconstruction, x, z_mu, z_var, z0, zk, ldj)
                #else:
                #    print(self.flavour, "is not a flavour, quiting.")
                #    exit()

                a_list.append(loss.cpu().data.numpy())

            # calculate max
            a_list += loss.view(-1, 1).detach().cpu().numpy()
            likelihood_x = logsumexp(a_list[-1])
            likelihood.append(likelihood_x - np.log(len(a_list[-1])))

        likelihood = np.array(likelihood)

        plot_histogram(-likelihood, self.model_history_path, mode)

        return -np.mean(likelihood)

    def display_reconstruction(self, epoch, data, reconstruction, display_rate=1):
        images_path = self.hparams_string + "/reconstruction/"
        create_missing_folders(images_path)
        x = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
        x_grid = tv.utils.make_grid(x)
        x_recon = reconstruction.view(-1, self.input_shape[0], self.input_shape[1],
                                      self.input_shape[2]).data
        x_recon_grid = tv.utils.make_grid(x_recon)

        if epoch % display_rate == 0:
            print("GENERATING RECONSTRUCTION IMAGES autoencoder!")
            tv.utils.save_image(x_grid, images_path + str(epoch) + "_original.png")
            tv.utils.save_image(x_recon_grid, images_path + str(epoch) + "_reconstruction_example.png")

    def prints(self):
        involment_df = pd.DataFrame(index=self.indices_names)
        print("Log file created: ",  "logs/" + self.__class__.__name__ + "_parameters.log")
        file_parameters = open("logs/" + self.__class__.__name__ + "_parameters.log", 'w+')
        #print("file:", file_parameters)
        print(*("LABELLED:", len(self.train_loader)), sep="\t", file=file_parameters)
        print("UNLABELLED:", len(self.train_loader_unlabelled), sep="\t", file=file_parameters)
        print("Number of classes:", self.num_classes, sep="\t", file=file_parameters)

        print("Total parameters:", self.get_n_params(), file=file_parameters)
        print("Total:", self.get_n_params(), file=file_parameters)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape, sep="\t", file=file_parameters)
        file_parameters.close()

        print("Log file created: ",  "logs/" + self.__class__.__name__ + "_involvment.log")
        file_involvment = open("logs/" + self.__class__.__name__ + "_involvment.log", 'w+')
        print("started", file=file_involvment)
        file_involvment.close()
        print("Log file created: ",  "logs/" + self.__class__.__name__ + ".log")
        file = open("logs/" + self.__class__.__name__ + ".log", 'w+')
        file.close()
        print("Labeled shape", len(self.train_loader))
        print("Unlabeled shape", len(self.train_loader_unlabelled))











