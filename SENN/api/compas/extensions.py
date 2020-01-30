import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from torch.utils.data import DataLoader


def get_sample(dataset, indx):
    x, t = dataset.__getitem__(indx)
    return x, t


def evaluate(model, dataset, print_freq=1000):
    correct = 0.
    # dataloader could be faster/more practical

    # faster: x_batch = Ctest.tensors[0] , t_batch = Ctest.tensors[1]
    batch_x = torch.zeros(len(dataset), len(get_sample(dataset, 0)[0]))
    batch_t = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        if print_freq != 0 and i % print_freq == 0:
            print(f"{i}/{len(dataset)}")
        x, t = get_sample(dataset, i)
        batch_x[i, :] = x
        batch_t[i] = t
    if print_freq != 0:
        print(f"{len(dataset)}/{len(dataset)}")
        print("Evaluating...")
    with torch.no_grad():
        batch_y = model(batch_x).squeeze()
        correct = torch.sum(torch.round(batch_y) == batch_t)
    acc = correct.type(torch.FloatTensor) / len(dataset)
    if print_freq != 0:
        print("accuracy = {:.3}".format(acc))

    return acc


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


def sample_local_lipschitz(model, dataset, mode=2, max_distance=None, top_k=1, cuda=False):
    """
        For every point in dataset, find pair point y in dataset that maximizes relative variation of model
            MODE 1:     || th(x) - th(y) ||/||x - y||
            MODE 2:     || th(x) - th(y) ||/||h(x) - h(y)||
            - dataset: a tds obkect
            - top_k : how many to return
            - max_distance: maximum distance between points to consider (radius)
    """
    model.eval()
    tol = 1e-10  # To avoid numerical problems

    # Create dataloader from tds without shuffle
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    n = len(dataset)  # len(dataset)
    Hs = []
    Ts = []

    for i, (inputs, targets) in enumerate(dataloader):
        # get the inputs
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            input_var = torch.tensor(inputs)
            target_var = torch.tensor(targets)

        _ = model(input_var)
        Ts.append(model.thetas.squeeze())
        Hs.append(model.concepts.squeeze())

    Ts = torch.cat(Ts, dim=0)
    num_dists = pairwise_distances(Ts)  # Numerator

    if mode == 1:
        denom_dists = pairwise_distances(dataset)
    if mode == 2:
        Hs = torch.cat(Hs)
        denom_dists = pairwise_distances(Hs)

    ratios = torch.Tensor(n, n)

    if max_distance is not None:
        # Distances above threshold: make them inf
        # print((denom_dists > max_distance).size())
        nonzero = torch.nonzero((denom_dists > max_distance).data).size(0)
        total = denom_dists.size(0) ** 2
        #         print('Number of zero denom distances: {} ({:4.2f}%)'.format(
        #                 total - nonzero, 100*(total-nonzero)/total))
        denom_dists[denom_dists > max_distance] = -1.0  # float('inf')
    # Same with self dists
    denom_dists[denom_dists == 0] = -1.0  # float('inf')
    ratios = (num_dists / denom_dists).data
    argmaxes = {k: [] for k in range(n)}
    vals, inds = ratios.topk(top_k, 1, True, True)
    argmaxes = {i: [(j, v) for (j, v) in zip(inds[i, :], vals[i, :])] for i in range(n)}
    return vals[:, 0].numpy(), argmaxes


def lipschitz_accuracy_plot(models, reg_lambdas, dataset, accuracies, logscale=True):
    # models: list of models with different reg_lambda
    # reg_lambda: use RegLambda
    # dataset: use COMPAS testset
    # accuracies: list of accs for each model
    # TO DO (not necessary): if accs not given, calculate them
    lips_list = []
    for i in range(len(reg_lambdas)):
        lips, argmaxes = sample_local_lipschitz(models[i], dataset, mode=2, top_k=1, max_distance=3)
        lips_list.append(lips)
    plt.title(r"Lipschitz estimate and accuracy versus $\lambda$", fontsize=12)
    plt.xlabel(r"Regularization Strength $\lambda$", fontsize=12)
    plt.ylabel("Local Lipschitz Estimate", fontsize=12)

    seaborn.boxplot(list(range(len(lips_list))), lips_list, palette="Set2", orient="v")
    plt.xticks(rotation=45)
    plt.xticks(range(len(accuracies)), ["{:0.0e}".format(x.value) for x in reg_lambdas])
    # plt.ylim(0,1)
    if logscale:
        plt.yscale("log")
    ax2 = plt.twinx()
    plt.tick_params(axis='y', colors='red')
    plt.plot(range(len(accuracies)), accuracies, "--", color="red")
    plt.scatter(range(len(accuracies)), [float(acc) for acc in accuracies], color="red")
    plt.ylabel("Prediction Accuracy", fontsize=12)
    ax2.yaxis.label.set_color('red')
    # plt.tight_layout()
    plt.show()
