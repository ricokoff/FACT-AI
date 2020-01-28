from api.load import RegLambda, load_compas
from api.utils import DATA_FOLDER
from robust_interpret.utils import lipschitz_feature_argmax_plot
import numpy as np
import seaborn
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
import os
from git import Repo

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch

from SENN.eval_utils import sample_local_lipschitz


def find_conflicting(df, labels, consensus_delta=0.2):
    """
        Find examples with same exact feat vector but different label.
        Finds pairs of examples in dataframe that differ only
        in a few feature values.

        Args:
            - differ_in: list of col names over which rows can differ
    """

    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in tqdm(range(len(df))):
        if full_dups[i] and (not i in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5) < consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                # print(scores.mean(), len(group))
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)


def load_compas_data(shuffle=False, batch_size=64):
    filename = DATA_FOLDER.joinpath('fairml', 'doc', 'example_notebooks', 'propublica_data_for_fairml.csv')
    print(filename)
    if not os.path.isfile(str(filename)):
        git_url = "https://github.com/adebayoj/fairml.git"
        repo_dir = "data/fairml"
        Repo.clone_from(git_url, repo_dir)

    df = pd.read_csv(filename)

    # Binarize num of priors var? Or normalize it 0,1?
    df['Number_of_Priors'] = np.sqrt(df['Number_of_Priors']) / (np.sqrt(38))
    compas_rating = df.score_factor.values  # This is the target??
    df = df.drop("score_factor", 1)

    pruned_df, pruned_rating = find_conflicting(df, compas_rating)
    x_train, x_test, y_train, y_test = train_test_split(pruned_df, pruned_rating, test_size=0.1, random_state=85)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=85)

    feat_names = list(x_train.columns)
    x_train = x_train.values  # pandas -> np
    x_test = x_test.values

    Tds = []
    Loaders = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler(with_std=False, with_mean=False)  # DOn't scale to make consitient with LIME/SHAP script
        transformed = scaler.fit_transform(foldx)
        # transformed = foldx
        tds = TensorDataset(torch.from_numpy(transformed).float(),
                            torch.from_numpy(foldy).view(-1, 1).float())
        loader = DataLoader(tds, batch_size=batch_size, shuffle=shuffle)
        Tds.append(tds)
        Loaders.append(loader)

    return (*Loaders, *Tds, df, feat_names)


def plot_lipschitz_feature(model, x):
    example = torch.tensor(list(x.values()))
    _, _, _, _, _, test, _, feat_names = load_compas_data()

    from pprint import pprint
    y = test.tensors[0][1]
    # y = np.random.choice(test.tensors[0][np.random.choice(np.arange(len(test.tensors[0])))].detach().numpy())

    lips, argmaxes = sample_local_lipschitz(model, test, mode=2, top_k=10, max_distance=3)
    max_lip = lips.max()
    imax = np.unravel_index(np.argmax(lips), lips.shape)[0]
    jmax = argmaxes[imax][0][0]
    argmax = test.tensors[0][jmax]
    argmax = y
    with torch.no_grad():
        pred_x = model(Variable(example.view(1, -1))).data
        att_x = model.thetas.data.squeeze().numpy().squeeze()

        # pred_argmax = model(Variable(argmax.view(1, -1))).data
        att_argmax = model.thetas.data.squeeze().numpy().squeeze()
    lipschitz_feature_argmax_plot(example, argmax, att_x, att_argmax,
                                  feat_names=feat_names,
                                  digits=2, figsize=(8, 8), widths=(2, 3))
# x = {
#     'Two_yr_Recidivism': 1.,
#      'Number_of_Priors': 1.,
#      'Age_Above_FourtyFive': 1.,
#      'Age_Below_TwentyFive':1.,
#      'African_American': 1.,
#      'Asian': 1.,
#      'Hispanic': 1.,
#      'Native_American': 1.,
#      'Other':1.,
#      'Female': 0.,
#      'Misdemeanor': 1.
#     }


# model = load_compas(RegLambda.E4)

# plot_lipschitz_feature(model, x)

def get_sample(dataset, indx):
    x, t = dataset.__getitem__(indx)
    return x, t

def evaluate(model, dataset, print_freq=1000, return_acc=False):
    model.eval()
    correct = 0.
    #dataloader could be faster/more practical
    
    #faster: x_batch = Ctest.tensors[0] , t_batch = Ctest.tensors[1]
    batch_x = torch.zeros(len(dataset),len(get_sample(dataset, 0)[0]))
    batch_t = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        if print_freq != 0 and i % print_freq == 0:
                print(f"{i}/{len(dataset)}")
        x, t = get_sample(dataset, i)
        batch_x[i,:] = x
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
    if return_acc:
        return acc
    return

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def sample_local_lipschitz(model, dataset, mode=2, max_distance = None, top_k=1, cuda = False):
    """
        For every point in dataset, find pair point y in dataset that maximizes relative variation of model
            MODE 1:     || th(x) - th(y) ||/||x - y||
            MODE 2:     || th(x) - th(y) ||/||h(x) - h(y)||
            - dataset: a tds obkect
            - top_k : how many to return
            - max_distance: maximum distance between points to consider (radius)
    """
    model.eval()
    tol = 1e-10 # To avoid numerical problems

    # Create dataloader from tds without shuffle
    dataloader = DataLoader(dataset, batch_size = 128, shuffle=False)
    n = len(dataset) # len(dataset)
    Hs = []
    Ts = []

    for i, (inputs, targets) in enumerate(dataloader):
        # get the inputs
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(targets)

        _ = model(input_var)
        Ts.append(model.thetas.squeeze())
        Hs.append(model.concepts.squeeze())

    Ts = torch.cat(Ts, dim = 0)
    num_dists = pairwise_distances(Ts) # Numerator

    if mode == 1:
        denom_dists = pairwise_distances(dataset)
    if mode == 2:
        Hs = torch.cat(Hs)
        denom_dists = pairwise_distances(Hs)

    ratios = torch.Tensor(n,n)

    if max_distance is not None:
        # Distances above threshold: make them inf
        #print((denom_dists > max_distance).size())
        nonzero = torch.nonzero((denom_dists > max_distance).data).size(0)
        total =  denom_dists.size(0)**2
#         print('Number of zero denom distances: {} ({:4.2f}%)'.format(
#                 total - nonzero, 100*(total-nonzero)/total))
        denom_dists[denom_dists > max_distance] = -1.0 #float('inf')
    # Same with self dists
    denom_dists[denom_dists == 0] = -1.0 #float('inf')
    ratios = (num_dists/denom_dists).data
    argmaxes = {k: [] for k in range(n)}
    vals, inds = ratios.topk(top_k, 1, True, True)
    argmaxes = {i:  [(j,v) for (j,v) in zip(inds[i,:],vals[i,:])] for i in range(n)}
    return vals[:,0].numpy(), argmaxes

def lipschitz_accuracy_plot(models, reg_lambdas, dataset, accuracies, logscale=True):
    #models: list of models with different reg_lambda
    #reg_lambda: use RegLambda
    #dataset: use COMPAS testset
    #accuracies: list of accs for each model
    #TO DO (not necessary): if accs not given, calculate them
    lips_list = []
    for i in range(len(reg_lambdas)):
        lips, argmaxes = sample_local_lipschitz(models[i], dataset, mode=2, top_k=1, max_distance=3)
        lips_list.append(lips)
    plt.title(r"Lipschitz estimate and accuracy versus $\lambda$", fontsize=12)
    plt.xlabel(r"Regularization Strength $\lambda$", fontsize=12)
    plt.ylabel("Local Lipschitz Estimate", fontsize=12)

    seaborn.boxplot(list(range(len(lips_list))),lips_list, palette="Set2", orient="v")
    plt.xticks(rotation=45)
    plt.xticks(range(len(accuracies)),["{:0.0e}".format(x.value) for x in reg_lambdas])
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
