import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from api.folders import COMPAS_IMAGES
from torch.utils.data import DataLoader

from api.datasets import COMPAS_TEST_SET


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def sample_local_lipschitz(model, mode=2, max_distance=None, top_k=1, cuda=False):
    """
        For every point in dataset, find pair point y in dataset that maximizes relative variation of model
            MODE 1:     || th(x) - th(y) ||/||x - y||
            MODE 2:     || th(x) - th(y) ||/||h(x) - h(y)||
            - top_k : how many to return
            - max_distance: maximum distance between points to consider (radius)
    """
    # Create dataloader from tds without shuffle
    dataloader = DataLoader(COMPAS_TEST_SET, batch_size=128, shuffle=False)
    n = len(COMPAS_TEST_SET)
    Hs = []
    Ts = []

    for i, (inputs, targets) in enumerate(dataloader):
        # get the inputs
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            input_var = torch.tensor(inputs)

        _ = model(input_var)
        Ts.append(model.thetas.squeeze())
        Hs.append(model.concepts.squeeze())

    Ts = torch.cat(Ts, dim=0)
    num_dists = pairwise_distances(Ts)  # Numerator

    if mode == 1:
        denom_dists = pairwise_distances(COMPAS_TEST_SET)
    if mode == 2:
        Hs = torch.cat(Hs)
        denom_dists = pairwise_distances(Hs)

    ratios = torch.Tensor(n, n)

    if max_distance is not None:
        denom_dists[denom_dists > max_distance] = -1.0
    # Same with self dists
    denom_dists[denom_dists == 0] = -1.0
    ratios = (num_dists / denom_dists).data
    vals, inds = ratios.topk(top_k, 1, True, True)
    argmaxes = {i: [(j, v) for (j, v) in zip(inds[i, :], vals[i, :])] for i in range(n)}
    return vals[:, 0].numpy(), argmaxes


def plot_lipschitz_accuracy(models, reg_lambdas, accuracies, logscale=True):
    """
    :param models: list of models with different reg_lambda
    :param reg_lambdas: use RegLambda
    :param accuracies: list of accs for each model
    :param logscale
    """
    lips_list = []
    for i in range(len(reg_lambdas)):
        lips, argmaxes = sample_local_lipschitz(models[i], mode=2, top_k=1, max_distance=3)
        lips_list.append(lips)
    plt.title(r"Lipschitz estimate and accuracy versus $\lambda$", fontsize=12)
    plt.xlabel(r"Regularization Strength $\lambda$", fontsize=12)
    plt.ylabel("Local Lipschitz Estimate", fontsize=12)

    seaborn.boxplot(list(range(len(lips_list))), lips_list, palette="Set2", orient="v")
    plt.xticks(rotation=45)
    plt.xticks(range(len(accuracies)), ["{:0.0e}".format(x.value) for x in reg_lambdas])
    if logscale:
        plt.yscale("log")
    ax2 = plt.twinx()
    plt.tick_params(axis='y', colors='red')
    plt.plot(range(len(accuracies)), accuracies, "--", color="red")
    plt.scatter(range(len(accuracies)), [float(acc) for acc in accuracies], color="red")
    plt.ylabel("Prediction Accuracy", fontsize=12)
    ax2.yaxis.label.set_color('red')
    plt.tight_layout()
    save_path = COMPAS_IMAGES.joinpath('lipschitz_stability.pdf')
    plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
