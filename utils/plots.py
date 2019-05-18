# normal standard curve

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from models.utils.utils import create_missing_folders
from scipy.stats import halfnorm
import pylab

def normal_curve(ax, mu=0., sigma=1.):
    try:
        sigma = np.sqrt(sigma)
    except:
        print("problem with sigma", sigma)
        return
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, mlab.normpdf(x, mu, sigma))
    plt.axvline(x=mu, c="r", linewidth=1)


def half_normal_curve(ax, mu=0., sigma=1., half_mu=0.):
    x = np.linspace(0, mu + 3*sigma, 100)
    # x = np.linspace(halfnorm.ppf(0.01),
    #                halfnorm.ppf(0.99), 100)
    ax.plot(x, halfnorm.pdf(x, mu, sigma))
    plt.axvline(x=mu, c="r", linewidth=1)
    plt.axvline(x=half_mu, c="g", linewidth=1)


def histograms_hidden_layers(xs, results_path, normalized, is_mean=True, epoch=0, depth=0, activated=False,
                             mu=None, var=None, axis=0, bins=50, flat=True, neuron=None):
    ax = plt.subplot(111)
    ax.set_xlabel("Hidden value")
    ax.set_ylabel("Frequency")
    plt.title("PDF of preactivation values")

    if neuron is None:
        neurons = "all"
    else:
        neurons = "single"
        xs = xs[:, neuron]

    if is_mean:
        xs = np.mean(xs, axis=axis)
    ax.hist(xs, bins=bins, alpha=0.5, density=True)

    if mu is None and var is None:
        mean_mean = float(np.mean(xs))
        mean_var = float(np.var(xs))
    elif mu is not None and var is not None:
        mean_mean = float(mu)
        mean_var = float(var)
    else:
        print("No images saved. Both mu and var must be either None or both numpy")
        return
    normal_curve(ax, mean_mean, mean_var)
    if activated:
        plt.axvline(x=float(np.mean(xs)), c="g", linewidth=1)

    #    half_normal_curve(ax, mu, var, float(np.mean(xs)))
    destination_folder_path = "/".join((results_path, "layers_histograms", "depth_"+str(depth),
                                        "activated_"+ str(activated), "normalized_"+str(normalized))) + "/"
    create_missing_folders(destination_folder_path)
    destination_file_path = destination_folder_path + "Hidden_values_hist_" + str(epoch) + "_activated"+ \
                            str(activated) + "_normalized" + str(normalized) + "_mean" + str(is_mean) + "_flat"\
                            + str(flat) + "_" + neurons + "neurons.png"
    plt.savefig(destination_file_path)
    plt.close()

def plot_performance(loss_total, accuracy, labels, results_path, filename="NoName", verbose=0, std_loss=None, std_accuracy=None):
    """

    :param loss_total:
    :param loss_labelled:
    :param loss_unlabelled:
    :param accuracy:
    :param labels:
    :param results_path:
    :param filename:
    :param verbose:
    :return:
    """
    fig2, ax21 = plt.subplots()
    n = list(range(len(accuracy["train"])))
    try:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:' + str(len(labels["valid"])))  # plotting t, a separately
        #ax21.plot(values["valid"], 'r-', label='Test:' + str(len(labels["valid"])))  # plotting t, a separately
    except:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:')  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["train"], yerr=[np.array(std_loss["train"]), np.array(std_loss["train"])],
                      c="b", label='Train')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["valid"], yerr=[np.array(std_loss["valid"]), np.array(std_loss["valid"])],
                      c="g", label='Valid')  # plotting t, a separately

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
    ax22.plot(accuracy["train"], 'c--', label='Train')  # plotting t, a separately
    ax22.plot(accuracy["valid"], 'k--', label='Valid')  # plotting t, a separately
    if std_accuracy is not None:
        ax22.errorbar(x=n, y=accuracy["train"], yerr=[np.array(std_accuracy["train"]), np.array(std_accuracy["train"])],
                      c="c", label='Train')  # plotting t, a separately
    if std_accuracy is not None:
        ax22.errorbar(x=n, y=accuracy["valid"], yerr=[np.array(std_accuracy["valid"]), np.array(std_accuracy["valid"])],
                      c="k", label='Valid')  # plotting t, a separately

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

