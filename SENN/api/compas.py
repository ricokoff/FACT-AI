from api.load import RegLambda, load_compas
from api.utils import DATA_FOLDER
from robust_interpret.utils import lipschitz_feature_argmax_plot
import numpy as np
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


def load_compas_data(shuffle=True, batch_size=64):
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
x = {
    'Two_yr_Recidivism': 1.,
     'Number_of_Priors': 1.,
     'Age_Above_FourtyFive': 1.,
     'Age_Below_TwentyFive':1.,
     'African_American': 1.,
     'Asian': 1.,
     'Hispanic': 1.,
     'Native_American': 1.,
     'Other':1.,
     'Female': 0.,
     'Misdemeanor': 1.
    }


model = load_compas(RegLambda.E4)

plot_lipschitz_feature(model, x)