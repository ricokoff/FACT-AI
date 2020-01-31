import os
import sys
from enum import auto, Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from git import Repo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


class RegLambda(Enum):
    ZERO = 0
    E4 = 1e-4
    E3 = 1e-3
    E2 = 1e-2
    E1 = 1e-1
    ONE = 1


class HType(Enum):
    CNN = auto()
    INPUT = auto()


class NConcepts(Enum):
    FIVE = 5
    TWENTY = 20


path = Path(__file__)

PROJECT_NAME = 'FACT-AI'

while path.name != PROJECT_NAME:
    path = path.parent

MODELS_FOLDER = path.joinpath('models')
DATA_FOLDER = path.joinpath('data')
IMAGES_FOLDER = path.joinpath('images')

for folder in [MODELS_FOLDER, DATA_FOLDER, IMAGES_FOLDER.joinpath('mnist'), IMAGES_FOLDER.joinpath('compas')]:
    if not os.path.isdir(folder):
        os.makedirs(folder)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def find_conflicting(df, labels, consensus_delta=0.2):
    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in tqdm(range(len(df))):
        if full_dups[i] and (i not in all_seen):
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
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)


def load_compas_data(shuffle=False, batch_size=64):
    filename = DATA_FOLDER.joinpath('fairml', 'propublica_data_for_fairml.csv')
    df = pd.read_csv(filename)

    df['Number_of_Priors'] = np.sqrt(df['Number_of_Priors']) / (np.sqrt(38))
    compas_rating = df.score_factor.values
    df = df.drop("score_factor", 1)

    pruned_df, pruned_rating = find_conflicting(df, compas_rating)
    x_train, x_test, y_train, y_test = train_test_split(pruned_df, pruned_rating, test_size=0.1, random_state=85)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=85)

    feat_names = list(x_train.columns)
    x_train = x_train.values
    x_test = x_test.values

    Tds = []
    Loaders = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler(with_std=False, with_mean=False)
        transformed = scaler.fit_transform(foldx)
        tds = TensorDataset(torch.from_numpy(transformed).float(),
                            torch.from_numpy(foldy).view(-1, 1).float())
        loader = DataLoader(tds, batch_size=batch_size, shuffle=shuffle)
        Tds.append(tds)
        Loaders.append(loader)

    return (*Loaders, *Tds, df, feat_names)


# COMPAS_TRAIN_LOADER, COMPAS_VALID_LOADER, COMPAS_TEST_LOADER, COMPAS_TRAIN_SET, COMPAS_VALID_SET, COMPAS_TEST_SET, \
# COMPAS_DATA_SET, COMPAS_FEAT_NAMES = load_compas_data()
_, _, _, COMPAS_TRAIN_SET, _, COMPAS_TEST_SET, _, COMPAS_FEAT_NAMES = load_compas_data()

MNIST_TRAIN_SET = MNIST(DATA_FOLDER.joinpath('MNIST'), train=True, download=True, transform=transform)
MNIST_TEST_SET = MNIST(DATA_FOLDER.joinpath('MNIST'), train=False, download=True, transform=transform)


def plot_accuracy_comparison(accuracies, titles):
    x = [l.value for l in RegLambda]
    xticks = np.linspace(0, 1, len(x))
    colors = ['C{}'.format(i) for i in range(8)]
    nrows = 2
    ncols = 2

    _, ax = plt.subplots(2, 2, figsize=(12, 6))

    for i in range(nrows):
        for j in range(ncols):
            y = list(accuracies[i + j * 2].values())
            ax[i, j].set(xlim=(0 - 0.1, 1 + 0.1), ylim=(0 - 0.1, 1 + 0.1))
            ax[i, j].set_xticks(xticks)
            ax[i, j].set_xticklabels([('{:0.0e}' if i != 0 and i != 1 else '{}').format(i) for i in x])
            ax[i, j].plot(xticks, y, 'go', alpha=0.9, color=colors[i + j * 2])
            ax[i, j].plot(xticks, y, '--', alpha=0.9, color=colors[i + j * 2 + 1])
            ax[i, j].set_title(titles[i + j * 2])
            ax[i, j].set_xlabel(r"$\lambda$")
            ax[i, j].set_ylabel("Accuracy")
            for n, m in zip(xticks, y):
                ax[i, j].text(n, m, "%.2f" % round(float(m), 2), va="bottom")
    plt.tight_layout()
    plt.suptitle("Accuracy", fontsize=16, y=1.05)
    plt.show()
