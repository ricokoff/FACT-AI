import sys

import torch
import matplotlib.pyplot as plt
import numpy as np

from api.load import load_compas, RegLambda
from api.utils import IMAGES_FOLDER, COMPAS_FEAT_NAMES


def plot_input_values(model, x, title='Input Value', ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots()

    example = torch.tensor(x)

    with torch.no_grad():
        prediction = model(example.view(1, -1).clone().detach()).data
    thetas = model.thetas.data.numpy().squeeze()

    red = '#FF4D4D'
    blue = '#3DE8F7'

    colors = [[blue, 'w'] if theta > sys.float_info.epsilon else [red, 'w'] for theta in thetas[:-1]]
    cells = [[label, value] for label, value in zip(COMPAS_FEAT_NAMES, x)]
    ax.axis('tight')
    ax.axis('off')

    ax.set_title(title, fontsize=16)
    ax.title.set_position([.5, 1.05])

    table = ax.table(cellText=cells, cellColours=colors, loc='center')
    table.auto_set_column_width(0)
    table.scale(1, 2)

    if show_and_save:
        plt.tight_layout()
        save_path = IMAGES_FOLDER.joinpath('compas', 'input_values.pdf'.format())
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_explanation(model, x, title='Explanation', show_labels=True, ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots()

    example = torch.tensor(x)

    with torch.no_grad():
        prediction = model(example.view(1, -1).clone().detach()).data
    thetas = model.thetas.data.numpy().squeeze()
    parameters = dict(zip(COMPAS_FEAT_NAMES, thetas))
    features = list(parameters.keys())
    values = list(parameters.values())
    maximum_value = np.absolute(np.array(values)).max()
    values = ((np.array(values) / maximum_value) * 100)
    index_sorted = range(len(values))[::-1]
    sorted_column_values = list(np.array(values)[index_sorted])

    yticks = np.arange(len(sorted_column_values)) + 0.5
    yticklabels = list(np.array(features)[index_sorted])

    red = '#FF4D4D'
    blue = '#3DE8F7'

    bar_colors = list(map(lambda value: blue if value > 0 else red, values))
    bar_colors = list(np.array(bar_colors))

    ax.set(xlim=(-100, 100))
    ax.set_xticks([-100, 0, 100])
    ax.set_xticklabels([-100, 0, 100])

    if show_labels:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticks([])

    ax.set_xlim(-105, 105)

    ax.set_title(title, fontsize=16)
    ax.title.set_position([.5, 1.05])
    ax.barh(yticks, sorted_column_values, align='center', color=bar_colors)

    if show_and_save:
        plt.tight_layout()
        save_path = IMAGES_FOLDER.joinpath('compas', 'explanations.pdf'.format())
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_input_values_regularized_unregularized_explanation(model, unregularized_model, xs, ax=None):
    show_and_save = False
    nrows = len(xs)
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))

    for i, x in enumerate(xs):
        print(i)
        plot_input_values(model, x, ax=ax[i, 0] if nrows > 1 else ax[0])
        plot_explanation(unregularized_model, x, title=r'Relevance Score $\theta$(x) (Scaled)',
                         ax=ax[i, 1] if nrows > 1 else ax[1])
        plot_explanation(model, x, title=r'Relevance Score $\theta$(x) (Scaled)', show_labels=False,
                         ax=ax[i, 2] if nrows > 1 else ax[2])

    if show_and_save:
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)
        save_path = IMAGES_FOLDER.joinpath('compas', 'input_values_regularized_unregularized_explanation.pdf')
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


x = [
    0.,  # Two_yr_Recidivism
    0.23,  # Number_of_Priors
    0.,  # Age_Above_FourtyFive
    1.,  # Age_Below_TwentyFive
    1.,  # African_American
    0.,  # Asian
    0.,  # Hispanic
    0.,  # Native_American
    0.,  # Other
    0.,  # Female
    0.,  # Misdemeanor
]

y = [
    0.,  # Two_yr_Recidivism
    0.23,  # Number_of_Priors
    0.,  # Age_Above_FourtyFive
    1.,  # Age_Below_TwentyFive
    0.,  # African_American
    0.,  # Asian
    0.,  # Hispanic
    0.,  # Native_American
    0.,  # Other
    0.,  # Female
    0.,  # Misdemeanor
]

model = load_compas(RegLambda.E2)
unregularized_model = load_compas(RegLambda.ZERO)
plot_input_values_regularized_unregularized_explanation(model, unregularized_model, [x, y])
