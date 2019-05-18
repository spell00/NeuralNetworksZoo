from __future__ import print_function
import time
import torch
from models.utils.utils import create_missing_folders
from models.NeuralNet import NeuralNet
from models.semi_supervised.utils.loss import mse_loss_function as calculate_losses
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
from models.utils.visual_evaluation import plot_histogram
from scipy.special import logsumexp
import torchvision as tv
from torch.nn import functional as F
from models.utils.distributions import log_gaussian, log_standard_gaussian
from scipy.stats import norm
from models.dimension_reduction.ordination import ordination2d
import pandas as pd

import torch.backends.cudnn as cudnn
if torch.cuda.is_available():
    cudnn.enabled = True
    device = torch.device('cuda:0')
else:
    cudnn.enabled = False
    device = torch.device('cpu')


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


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g


class AE(NeuralNet):

    def __init__(self):
        super(AE, self).__init__()
        self.pretrained = False
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.mom = None
        self.lr = None
        self.batch_size = None
        self.latent_dim = None
        self.epsilon_std = None
        self.model_file_name = None
        self.input_size = None
        self.warmup = None
        self.early_stopping = None
        self.number_combination = None
        self.z = []
        self.zs_train = []
        self.as_train = []
        self.zs_test = []
        self.as_test = []
        self.zs_train_targets = []
        self.zs_test_targets = []
        self.n_layers = None
        self.n_hidden = None
        self.input_size = None

        self.encoder_pre = None
        self.encoder_gate = None
        self.q_z_mean = None
        self.q_z_log_var = None

        # decoder: p(x | z)
        self.decoder_pre = None
        self.decoder_gate = None
        self.reconstruction = None

        self.sigmoid = None
        self.tanh = None

        self.Gate = None

        self.encoder_pre = None
        self.encoder_gate = None
        self.bn_encoder = None

        self.decoder_pre = None
        self.decoder_gate = None
        self.bn_decoder = None
        self.ladder = None
        self.flavour = None
        self.z_dim_last = None
        self.optim_type = None

        self.best_loss = -1

        self.reconstruction_function = nn.MSELoss(size_average=False, reduce=False)


    def define_configurations(self, flavour, early_stopping=100, warmup=100, ladder=True, z_dim=40, epsilon_std=1.0,
                              model_name="vae", init="glorot", optim_type="adam", auxiliary=True, supervised="no",
                              l1=0., l2=0.):
        import os
        self.l1 = l1
        self.l2 = l2

        self.ladder = ladder
        self.supervised = supervised
        self.auxiliary = auxiliary
        self.flavour = flavour
        self.epsilon_std = epsilon_std
        self.warmup = warmup
        self.early_stopping = early_stopping
        self.init = init
        self.z_dim_last = z_dim
        self.set_init(init)
        self.optim_type = optim_type
        self.set_optim(optim_type)

        print('create model')
        # importing model
        self.model_file_name = rename_model(model_name, warmup, z_dim, l1=self.l1, l2=self.l2)
        self.hparams_string = "/".join([os.getcwd(), "results", model_name, "num_elements"+str(self.num_elements),
                                        "n_flows"+str(self.n_flows),"z_dim"+str(self.z_dim_last),
                                        "lr"+str(self.lr), "ladder"+str(self.ladder), self.flavour])

        if self.has_cuda:
            self.cuda()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def run(self, epochs=10, clip_grad=0, gen_rate=10, lambda1=0, lambda2=0):
        best_loss = 100000.
        e = 0
        self.train_loss_history = []
        self.train_rec_history = []
        self.train_kl_history = []

        self.val_loss_history = []
        self.val_rec_history = []
        self.val_kl_history = []

        time_history = []
        self.optimizer = self.optimization(self.parameters(), lr=float(self.lr), weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True,
                                                                    cooldown=0, patience=100)

        if self.has_cuda:
            self.cuda()

        for epoch in range(self.epoch, epochs + 1):
            self.epoch += 1
            self.optimizer.zero_grad()
            self.zs_train = []
            self.zs_train_targets = []
            self.zs_test = []
            self.zs_test_targets = []
            time_start = time.time()
            train_loss_epoch, train_rec_epoch, train_kl_epoch = self.train_vae(epoch, clip_grad)
            self.kill_dead_neurites(threshold=1e-8, sparse_tensor=False, verbose=True)
            val_loss_epoch, val_rec_epoch, val_kl_epoch = self.evaluate_vae(mode='validation')
            self.scheduler.step(val_loss_epoch)
            time_end = time.time()

            time_elapsed = time_end - time_start

            # appending history
            self.train_loss_history.append(train_loss_epoch)
            self.train_rec_history.append(train_rec_epoch)
            self.train_kl_history.append(train_kl_epoch)
            self.val_loss_history.append(val_loss_epoch)
            self.val_rec_history.append(val_rec_epoch)
            self.val_kl_history.append(val_kl_epoch)
            time_history.append(time_elapsed)

            # printing results
            print('Epoch: {}/{}, Time elapsed: {:.8f}s\n'
                  '* Train loss: {:.8f}   (re: {:.8f}, kl: {:.8f})\n'
                  'o Val.  loss: {:.8f}   (re: {:.8f}, kl: {:.8f})\n'
                  '--> Early stopping: {}/{} (BEST: {:.8f})\n'.format(
                self.epoch, epochs, time_elapsed,
                train_loss_epoch, train_rec_epoch, train_kl_epoch,
                val_loss_epoch, val_rec_epoch, val_kl_epoch,
                e, self.early_stopping, best_loss
            ))

            # early-stopping
            if val_loss_epoch < best_loss:
                e = 0
                best_loss = val_loss_epoch
                print("Saving model...")
                self.save_model()
                print("Done")
            else:
                e += 1
                if e > self.early_stopping:
                    break

            if epoch < self.warmup:
                e = 0
            if self.z_dim_last == 2:
                self.plot_z()

            del val_loss_epoch, val_rec_epoch, val_kl_epoch, train_loss_epoch, train_rec_epoch, train_kl_epoch
        del best_loss
        # FINAL EVALUATION
        self.test_loss, self.test_re, self.test_kl, self.test_log_likelihood, self.train_log_likelihood, \
            self.test_elbo, self.train_elbo = self.evaluate_vae(mode='valid', gen_rate=gen_rate)
        self.print_final()

    def train_vae(self, epoch, clip_grad):
        # set loss to 0
        train_loss = 0
        train_rec = 0
        train_kl = 0
        # set model in training mode
        self.train()

        # start training
        if self.warmup == 0:
            beta = 1.
        else:
            beta = 1. * (epoch - 1) / self.warmup
            if beta > 1.:
                beta = 1.
        print('beta: {}'.format(beta))
        print("Labelled", len(self.train_loader))
        data = None
        reconstruction = None
        # TODO add n/a data here?

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.view(-1, self.input_size)
            if self.has_cuda:
                data, target = data.cuda(), target.cuda()

            loss, rec, kl, reconstruction, z_q, a_q = self.calculate_losses(data, beta=1., likelihood=F.mse_loss)
            if type(z_q) == dict:
                z_q = z_q[-1]

            self.zs_train += [z_q]
            if a_q is not None:
                self.as_train += [a_q]
            self.zs_train_targets += [target]
            # backward pass
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            else:
                pass

            self.optimizer.step()
            self.optimizer.zero_grad()
            # optimization
            train_loss += loss.item()

            train_rec += rec.item()
            train_kl += kl.item()

            del rec, kl, target, z_q, loss

        self.zs_train = torch.stack(self.zs_train).detach().cpu().numpy()
        self.zs_train = np.vstack(self.zs_train)
        self.zs_train_targets = torch.stack(self.zs_train_targets).detach().cpu().numpy()
        self.zs_train_targets = np.vstack(self.zs_train_targets)
        cat_targets = np.argmax(self.zs_train_targets, axis=1)
        data_frame_train = pd.DataFrame(self.zs_train)
        ordination2d(data_frame_train, "pca", self.hparams_string + "/pca/", self.dataset_name, epoch, targets=cat_targets)
        ordination2d(data_frame_train, "lda", self.hparams_string + "/lda/", self.dataset_name, epoch, targets=cat_targets)

        print("Generating images")
        # if self.epoch % gen_rate == 0:
        del reconstruction, data
        # TODO address this in a proper way
        try:
            print("Unlabelled:", len(self.train_loader_unlabelled))
            for batch_idx, (data, _) in enumerate(self.train_loader_unlabelled):
                data = data.view(-1, self.input_size)
                if self.has_cuda:
                    data = data.cuda()

                loss, rec, kl, reconstruction, z_q = self.calculate_losses(data, beta=1., likelihood=F.mse_loss)

                # backward pass
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                else:
                    pass

                self.optimizer.step()
                # optimization
                train_loss += loss.item()

                train_rec += rec.item()
                train_kl += kl.item()

                del rec, kl, data, reconstruction, loss, clip_grad
        except:
            pass

        # calculate final loss
        train_loss /= len(self.train_loader)  # loss function already averages over batch size
        train_rec /= len(self.train_loader)  # rec already averages over batch size
        train_kl /= len(self.train_loader)  # kl already averages over batch size
        return train_loss, train_rec, train_kl

    def evaluate_vae(self, mode="validation", calculate_likelihood=True, gen_rate=10):
        # set loss to 0
        data, reconstruction = None, None

        with torch.no_grad():
            print("EVALUATION!")
            log_likelihood_test, log_likelihood_train, elbo_test, elbo_train = None, None, None, None
            evaluate_loss = 0
            evaluate_rec = 0
            evaluate_kl = 0
            # set model to evaluation mode

            if mode == "validation":
                data_loader = self.valid_loader
            elif mode == "valid":
                data_loader = self.test_loader

            if torch.cuda.is_available:
                has_cuda = True

            # evaluate
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.view(-1, self.input_size)
                if self.has_cuda:
                    data, target = data.cuda(), target.cuda()

                loss, rec, kl, reconstruction, z_q_test, a_q_test = self.calculate_losses(data, beta=1., likelihood=F.mse_loss)
                if type(z_q_test) == dict:
                    z_q_test = z_q_test[-1]
                    self.zs_test += [z_q_test]

                if type(a_q_test) == dict:
                    a_q_test = a_q_test[-1]
                    self.as_test += [a_q_test]
                self.zs_test_targets += [target]

                evaluate_loss += loss.item()
                evaluate_rec += rec.item()
                evaluate_kl += kl.item()
                del loss, kl, target, rec, z_q_test, a_q_test
            if mode == 'valid':
                # load all data
                test_data = Variable(data_loader.dataset.parent_ds.test_data)
                test_data = test_data.view(-1, np.prod(np.array(self.input_shape)))
                test_target = Variable(data_loader.dataset.parent_ds.test_labels)
                full_data = Variable(self.train_loader.dataset.train_data)


                if has_cuda:
                    test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

                full_data = full_data.data.cpu().float().cuda()
                test_data = test_data.data.cpu().float().cuda()
                #full_data = torch.bernoulli(full_data)
                #test_data = torch.bernoulli(test_data)

                full_data = Variable(full_data.double(), requires_grad=False)

                # VISUALIZATION: plot reconstructions
                (_, z_mean_recon, z_logvar_recon), _ = self.encoder(test_data)
                z_recon = self.reparameterize(z_mean_recon, z_logvar_recon)


                # VISUALIZATION: plot generations
                z_sample_rand = Variable(torch.normal(torch.from_numpy(np.zeros((25, self.z_dim_last))).float(), 1.))
                if has_cuda:
                    z_sample_rand = z_sample_rand.cuda()

                full_data = full_data.data.cpu().float().cuda()
                test_data = test_data.data.cpu().float().cuda()
                elbo_test, elbo_train = self.calculate_elbo(full_data, test_data)
                if calculate_likelihood:
                    elbo_test, elbo_train, log_likelihood_test, log_likelihood_train = \
                        self.calculate_likelihood(full_data, test_data)

            # calculate final loss
            evaluate_loss /= len(data_loader)  # loss function already averages over batch size
            evaluate_rec /= len(data_loader)  # rec already averages over batch size
            evaluate_kl /= len(data_loader)  # kl already averages over batch size
            self.generate_random()
            self.generate_uniform_gaussian_percentiles()
            self.display_reconstruction(data, reconstruction)
            del data, reconstruction
            if mode == 'valid':
                return evaluate_loss, evaluate_rec, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
            else:
                if evaluate_loss < self.best_loss or self.best_loss == -1:
                    print("BEST EVALUATION LOSS: SAVING MODEL")
                    #self.best_loss = evaluate_loss
                    #self.save_model()

                return evaluate_loss, evaluate_rec, evaluate_kl

    def calculate_losses(self, data, beta=1., likelihood=F.mse_loss, ys=None):
        z_q = None
        if self.ladder:
            ladder = "ladder"
        else:
            ladder = "not_ladder"
        data = torch.tanh(data)
        if self.flow_type in ["o-sylvester", "t-sylvester", "h-sylvester"] and not self.ladder:
            z_q = {0: None, 1: None}
            reconstruction, mu, log_var, self.log_det_j, z_q[0], z_q[-1] = self.run_sylvester(data, y=ys, exception=True, auxiliary=self.auxiliary)
            log_p_zk = log_standard_gaussian(z_q[-1])
            # ln q(z_0)  (not averaged)
            # mu, log_var, r1, r2, q, b = q_param_inverse
            log_q_zk = log_gaussian(z_q[0], mu, log_var=log_var) - self.log_det_j
            # N E_q0[ ln q(z_0) - ln p(z_k) ]
            self.kl_divergence = log_q_zk - log_p_zk
            del log_q_zk, log_p_zk
        else:
            try:
                reconstruction, z_q = self(data, ys)
            except:
                reconstruction = self(data, ys)


        kl = beta * self.kl_divergence

        likelihood = torch.sum(likelihood(reconstruction, data.float(), reduce=False), dim=-1)

        if self.ladder:
            params = torch.cat([x.view(-1) for x in self.reconstruction.parameters()])
        else:
            params = torch.cat([x.view(-1) for x in self.decoder.reconstruction.parameters()])

        l1_regularization = self.l1 * torch.norm(params, 1).cuda()
        l2_regularization = self.l2 * torch.norm(params, 2).cuda()
        try:
            assert l1_regularization >= 0. and l2_regularization >= 0.
        except:
            print(l1_regularization, l2_regularization)
        loss = torch.mean(likelihood + kl.cuda() + l1_regularization + l2_regularization)

        del data, params, l1_regularization, l2_regularization

        return loss, torch.mean(likelihood), torch.mean(kl), reconstruction, z_q, None

    def plot_z(self):
        # some torch Tensors returned non-cuda error... trying easiest fix
        self.cuda()
        pos = None
        i = None
        label = None
        zs_test = None
        as_test = None
        as_train = None
        labs_test = None
        if type(self.zs_train) is list:
            try:
                zs_train = np.vstack(torch.stack(self.zs_train).detach().cpu().numpy())
                as_train = np.vstack(torch.stack(self.as_train).detach().cpu().numpy())
            except:
                try:
                    zs_train = np.vstack(np.vstack(torch.stack(self.zs_train).detach().cpu().numpy()))
                    as_train = np.vstack(np.vstack(torch.stack(self.as_train).detach().cpu().numpy()))
                except:
                    zs_train = np.vstack(self.zs_train)
                    if len(self.as_train) > 0:
                        as_train = np.vstack(self.as_train)

            try:
                labs_train = np.argmax(np.vstack(torch.stack(self.zs_train_targets).detach().cpu().numpy()), 1)
            except:
                labs_train = np.array(self.zs_train_targets)
        else:
            try:
                zs_train = self.zs_train.detach().cpu().numpy()
                as_train = self.as_train.detach().cpu().numpy()
            except:
                zs_train = self.zs_train
                as_train = self.as_train
            labs_train = self.zs_train_targets


        # If their is more than one batch (should be the case), than I remove the last mini-batch if it is not the same
        # size as the rest. Reason: vstack not working otherwise... not best option IMO, but for vizualisation purposes
        # it should not be a big deal

        if len(self.zs_test) > 1:
            if len(self.zs_test[-1]) is not len(self.zs_test[0]):
                self.zs_test = self.zs_test[:-1]
                self.zs_test_targets = self.zs_test_targets[:-1]
        if len(self.zs_test) > 0:
            if type(self.zs_test) is list:
                try:
                    zs_test = np.vstack(torch.stack(self.zs_test).detach().cpu().numpy())
                    if len(self.as_test) > 0:
                        as_test = np.vstack(torch.stack(self.as_test).detach().cpu().numpy())
                except:
                    try:
                        zs_test = np.vstack(np.vstack(torch.stack(self.zs_test).detach().cpu().numpy()))
                        if len(self.as_test) > 0:
                            as_test = np.vstack(np.vstack(torch.stack(self.as_test).detach().cpu().numpy()))
                    except:
                        zs_test = np.vstack(self.zs_test)
                        if len(self.as_test) > 0:
                            as_test = np.vstack(self.as_test)
                try:
                    labs_test = np.argmax(np.vstack(torch.stack(self.zs_test_targets).detach().cpu().numpy()), 1)
                except:
                    labs_test = self.zs_test_targets
            else:
                try:
                    zs_test = self.zs_test.detach().cpu().numpy()
                    if len(self.as_test) > 0:
                        as_test = self.as_test.detach().cpu().numpy()
                except:
                    zs_test = self.zs_test
                    if len(self.as_test) > 0:
                        as_test = self.as_test

                labs_test = self.zs_test_targets

        if zs_train.shape[1] == 2:
            self.plot_latent(zs_train, labs=labs_train, latent_type="z", step="train", generated=False)
            try:
                self.plot_latent(zs_test, labs=labs_test, latent_type="z", step="test", generated=False)
            except:
                print("Test not plotted")
        try:
            self.plot_latent(as_train, labs=labs_train, latent_type="a", step="train", generated=False)
        except:
            pass
        try:
            self.plot_latent(as_test, labs=labs_test, latent_type="a", step="test", generated=False)
        except:
            pass
        del zs_train, zs_test, labs_train, labs_test, label, pos, as_train, as_test

    def plot_latent(self, zs, labs, latent_type, generated, step, max_samples=1000):
        fig, ax = plt.subplots()  # create figure and axis
        if type(labs) is not list:
            if len(labs.shape) > 1:
                labs = [x.tolist().index(1) for x in labs]
        for i, label in enumerate(self.labels_set):
            if label == "N/A":
                continue
            pos1 = np.array([l for l, x in enumerate(labs) if str(x) == str(label)])
            # np.random.shuffle(pos)
            # pos2 = np.array(pos1[:max_samples], dtype=int)
            try:
                ax.scatter(zs[pos1, 0], zs[pos1, 1], s=3, marker='.', label=str(label))
            except:
                zs = np.vstack(zs)
                ax.scatter(zs[pos1, 0], zs[pos1, 1], s=3, marker='.', label=str(label))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.tight_layout()
        #fig.tight_layout()
        new_string = "/".join([self.hparams_string, latent_type, step, "generated:"+str(generated)])
        print("Plotting", latent_type, " at:\n", new_string, "\n")
        create_missing_folders(new_string)
        fig.savefig(new_string + "/" + str(self.epoch))
        plt.close(fig)


    def calculate_elbo(self, test_data, full_data):
        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = self.calculate_lower_bound(test_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = self.calculate_lower_bound(full_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))
        return elbo_test, elbo_train

    def load_ae(self, load_history=False):
        print("LOADING PREVIOUSLY TRAINED autoencoder")
        trained_vae = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.state_dict')
        shape_w0 = trained_vae["encoder.gates_layers.0.weight"].shape
        shape_z = trained_vae["decoder.hidden.0.weight"].shape
        in_features_w0 = self.encoder.gates_layers[0].in_features
        z_to_add = self.num_classes

        # The 4 following torch.cat are to make trained_vae the correct size for load_state_dict
        # TODO change 0s for glorot init...
        # self.encoder.gates_layers[0].weights
        trained_vae["encoder.gates_layers.0.weight"] = torch.cat((trained_vae["encoder.gates_layers.0.weight"],
                                                            torch.nn.init.kaiming_normal_(torch.zeros((shape_w0[0], in_features_w0 - shape_w0[1]))).cuda()),1)
        trained_vae["encoder.hidden.0.weight"] = torch.cat((trained_vae["encoder.hidden.0.weight"],
                                                            torch.nn.init.kaiming_normal_(torch.zeros((shape_w0[0], in_features_w0 - shape_w0[1]))).cuda()),1)
        trained_vae["decoder.hidden.0.weight"] = torch.cat((trained_vae["decoder.hidden.0.weight"],
                                                            torch.nn.init.kaiming_normal_(torch.zeros((shape_z[0], z_to_add)).cuda())), 1)
        trained_vae["decoder.gates_layers.0.weight"] = torch.cat((trained_vae["decoder.gates_layers.0.weight"],
                                                            torch.nn.init.kaiming_normal_(torch.zeros((shape_z[0], z_to_add)).cuda())), 1)

        # strict is False to load even the elements not initially in the current state.
        self.load_state_dict(trained_vae, strict=False)
        self.pretrained = True

        # The previous 4 modification of trained_vae and stract=False are necessary to load a pre-trained VAE
        # The pretrained VAE (M1) has to correspond to the VAE in the M2-VAE (some-hyper-parameters included,
        # though this could be modified so the hyper-parameters don't have to be the same (e.g. the warm-up currently
        # has to be the same because of the way it is saved and the name required to load it, however the
        # warm-up in the pre-training should not have to be the same than in the M2 part))


        if load_history:
            self.epoch = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
            self.train_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_loss')
            self.train_rec_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_re')
            self.train_kl_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kl')
            self.val_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_loss')
            self.val_rec_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_re')
            self.val_kl_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_kl')

    def load_classifier(self):
        trained_classifier = torch.load(
            self.model_history_path + self.flavour + "_" + self.model_file_name + 'classifier.state_dict')
        self.classifier = self.load_state_dict(trained_classifier)

    def save_model(self):
        # SAVING
        print("MODEL SAVED AT LOCATION:", self.model_history_path)
        create_missing_folders(self.model_history_path)
        torch.save(self.state_dict(), self.model_history_path + self.flavour + "_" + self.model_file_name +'.state_dict')
        if self.supervised == "semi":
            torch.save(self.classifier.state_dict(),  self.model_history_path + self.flavour + "_" + self.model_file_name +'classifier.state_dict')
        torch.save(self.train_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_loss')
        torch.save(self.train_rec_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_re')
        torch.save(self.train_kl_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kl')
        torch.save(self.val_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_loss')
        torch.save(self.val_rec_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_re')
        torch.save(self.val_kl_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_kl')
        torch.save(self.epoch, self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
        #torch.save(self.test_log_likelihood, self.model_history_path + self.flavour + '.test_log_likelihood')
        #torch.save(self.test_loss, self.model_history_path + self.flavour + '.test_loss')
        #torch.save(self.test_re, self.model_history_path + self.flavour + '.test_re')
        #torch.save(self.test_kl, self.model_history_path + self.flavour + '.test_kl')

        # TODO SAVE images of 100 generated images

    def print_final(self, calculate_likelihood=True):
        print('FINAL EVALUATION ON TEST SET\n'
              'ELBO (TEST): {:.2f}\n'
              'ELBO (TRAIN): {:.2f}\n'
              'Loss: {:.2f}\n'
              're: {:.2f}\n'
              'kl: {:.2f}'.format(self.test_elbo, self.train_elbo, self.test_loss, self.test_re, self.test_kl))

        if calculate_likelihood:
            print('FINAL EVALUATION ON TEST SET\n'
                  'LogL (TEST): {:.2f}\n'
                  'LogL (TRAIN): {:.2f}'.format(self.test_log_likelihood, self.train_log_likelihood))

        with open(self.model_history_path + 'vae_experiment_log.txt', 'a') as f:
            print('FINAL EVALUATION ON TEST SET\n'
                  'ELBO (TEST): {:.2f}\n'
                  'ELBO (TRAIN): {:.2f}\n'
                  'Loss: {:.2f}\n'
                  're: {:.2f}\n'
                  'kl: {:.2f}'.format(self.test_elbo, self.train_elbo, self.test_loss, self.test_re, self.test_kl), file=f)

            if calculate_likelihood:
                print('FINAL EVALUATION ON TEST SET\n'
                      'LogL (TEST): {:.2f}\n'
                      'LogL (TRAIN): {:.2f}'.format(self.test_log_likelihood, self.train_log_likelihood), file=f)

    def save_config(self):
        pass

    def calculate_likelihood(self, data, mode='valid', s=5000, display_rate=100):
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
            if j % display_rate == 0:
                print("\revaluating likelihood:", j, "/", n_test, -np.mean(likelihood), end="", flush=True)
            # Take x*                    print("\rProgress: {:.2f}%".format(progress), end="", flush=True)
            x_single = data[j].unsqueeze(0).view(self.input_shape[0], self.input_size)
            a = []
            for _ in range(0, int(r)):
                # Repeat it for all training points
                x = x_single.expand(s, x_single.size(1))

                # pass through VAE
                #if self.flavour in ["ccLinIAF", "hf", "vanilla", "normflow"]:
                loss, rec, kl, _, _ = self.calculate_losses(x)
                #elif self.flavour in ["o-sylvester", "h-sylvester", "t-sylvester"]:
                #    reconstruction, z_mu, z_var, ldj, z0, zk = self.run_sylvester(x, auxiliary=False)
                #    loss, rec, kl = calculate_losses(reconstruction, x, z_mu, z_var, z0, zk, ldj)
                #else:
                #    print(self.flavour, "is not a flavour, quiting.")
                #    exit()

                a.append(loss.cpu().data.numpy())

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0], 1))
            likelihood_x = logsumexp(a)
            likelihood.append(likelihood_x - np.log(len(a)))

        likelihood = np.array(likelihood)

        plot_histogram(-likelihood, self.model_history_path, mode)

        return -np.mean(likelihood)

    def calculate_lower_bound(self, x_full):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        loss = torch.Tensor([])
        mb = 500

        for i in range(int(x_full.size(0) / mb)):

            x = x_full[i * mb: (i + 1) * mb].view(-1, self.input_size)

            if self.flavour in ["ccLinIAF", "hf", "vanilla", "normflow"]:
                loss, _, _, _, _ = self.calculate_losses(x)
            elif self.flavour in ["o-sylvester", "h-sylvester", "t-sylvester"]:
                reconstruction, z_mu, z_var, ldj, z0, zk = self.run_sylvester(x, torch.Tensor([]).cuda(),
                                                                              torch.Tensor([]).cuda(), auxiliary=False)
                loss, _, _ = calculate_losses(reconstruction, x, z_mu, z_var, z0, zk, ldj)
            else:
                print(self.flavour, "is not a flavour, quiting.")
                exit()

            # CALCULATE LOWER-BOUND: re + kl - ln(N)
            lower_bound += loss.cpu().item()

        lower_bound = lower_bound / x_full.size(0)

        return lower_bound

    def plot_z_stats(self, z, path, generate="generated", max=5000):
        fig, ax = plt.subplots()  # create figure and axis
        plt.boxplot(z)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.tight_layout()
        fig.tight_layout()
        path = "/".join([path, "plots/vae_z_stats", generate]) + "/"
        create_missing_folders(path)
        fig.savefig(path + self.flavour + "_" + str(self.epoch) + '_lr' + str(self.lr) + '_bs'+str(self.batch_size) + ".png")
        plt.close(fig)

        del z, path, generate

    def generate_random(self, max=1000):
        self.eval()
        print("GENERATING RANDOM IMAGES autoencoder!")
        images_path = self.hparams_string + "/generated_random/"
        create_missing_folders(images_path)

        rand_z = torch.randn(self.batch_size, self.z_dim_last).cuda()
        self.plot_z_stats(rand_z.detach().cpu().numpy(), generate="/random_generated/" + self.prior_dist + "/", path=images_path, max=max)
        new_x = self.sample(rand_z)
        if len(self.input_shape) > 1:
            images = new_x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
            images_grid = tv.utils.make_grid(images)
            print("Images location:", images_path)
            tv.utils.save_image(images_grid, images_path + str(self.epoch) + self.dataset_name + "generated.png")
            del images_grid, images
        del rand_z, new_x, images_path

    def generate_uniform_gaussian_percentiles(self, n=20, verbose=1, max=1000):
        self.eval()
        print("GENERATING gaussian percentiles IMAGES autoencoder!")

        xs_grid = torch.Tensor(np.vstack([np.linspace(norm.ppf(0.01), norm.ppf(0.99), n**2) for _ in range(self.z_dim_last)]).T)

        this_path = self.hparams_string + "/gaussian_percentiles/"
        if verbose > 0:
            print("GENERATING SS DGM IMAGES AT", this_path)

        print("image path:", this_path)
        create_missing_folders(this_path)
        grid = torch.Tensor(xs_grid).to(device)
        if self.z_dim_last == 2:
            self.plot_z_stats(xs_grid, generate="/ugp_generated/", path=this_path, max=max)

        try:
            new_x = torch.stack([self.sample(g.view(1, -1)) for g in grid])
        except:
            new_x = self.sample(grid)
        if len(self.input_shape) > 1:
            images = new_x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data

            assert n == int(images.shape[0]) / n
            images_grid = tv.utils.make_grid(images, int(np.sqrt(images.shape[0])))

            create_missing_folders(this_path)
            tv.utils.save_image(images_grid, this_path + str(self.epoch) + self.dataset_name + "gaussian_uniform_generated.png")
            del images_grid, images, new_x, xs_grid

    def display_reconstruction(self, data, reconstruction):
        self.eval()
        print("GENERATING RECONSTRUCTION IMAGES autoencoder!")
        hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                                   "z_dim"+str(self.z_dim_last), "unsupervised", "lr"+str(self.lr),
                                   "ladder"+str(self.ladder), self.flavour])
        x = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
        x_grid = tv.utils.make_grid(x)
        x_recon = reconstruction.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
        x_recon_grid = tv.utils.make_grid(x_recon)
        images_path = self.hparams_string + "/recon/"
        print("Images location:", images_path)

        create_missing_folders(images_path)
        tv.utils.save_image(x_grid, images_path + "original_" + str(self.epoch) + ".png")
        tv.utils.save_image(x_recon_grid, images_path + "reconstruction_example_" + str(self.epoch) + ".png")

    def forward(self, *args):
        print("Nothing going on in forward of autoencoder.py")
        exit()


