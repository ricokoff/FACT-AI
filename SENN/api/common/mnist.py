import matplotlib.pyplot as plt
import numpy as np
import torch

from api.datasets import MNIST_TEST_SET
from api.folders import MNIST_IMAGES


def get_digit(index):
    d, t = MNIST_TEST_SET.__getitem__(index)
    return d.view(28, 28), t


def plot_digit(x, label, title='Original', ax=None, fs=10):
    show_and_save = False
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots()

    x = np.flipud(x.squeeze().numpy())
    x_size, y_size = x.shape

    ax.set(xlim=(0, x_size - 1), ylim=(0, y_size - 1))
    ax.set_xticks([0, x_size - 1])
    ax.set_xticklabels([0, x_size], fontsize=fs)
    ax.xaxis.tick_top()
    ax.xaxis.set_tick_params(labeltop='on', labelbottom='off')
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks([y_size - 1, 0])
    ax.set_yticklabels([0, y_size], fontsize=fs)

    ax.set_title(title, fontsize=fs)
    ax.title.set_position([.5, 1.05])
    ax.imshow(x)

    if show_and_save:
        plt.tight_layout()
        plt.subplots_adjust(bottom=0)
        save_path = MNIST_IMAGES.joinpath('digit_{}.pdf'.format(label))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_activation(model, x, label, title='SENN', ax=None, fs=10):
    show_and_save = False
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots()

    with torch.no_grad():
        prediction = model(x.reshape(1, 1, 28, 28)).max(1)[1]
    theta = model.thetas.data.numpy().squeeze()
    concept_names = ['C{}'.format(i + 1) for i in range(theta.shape[0])]
    concept_values = theta[:, prediction].squeeze()
    concepts = dict(zip(concept_names, concept_values))
    columns = list(concepts.keys())
    values = list(concepts.values())
    maximum_value = np.absolute(np.array(values)).max()
    values = ((np.array(values) / maximum_value) * 100)
    index_sorted = range(len(values))[::-1]
    sorted_column_values = list(np.array(values)[index_sorted])

    yticks = np.arange(len(sorted_column_values)) + 0.5
    yticklabels = list(np.array(columns)[index_sorted])

    red = '#FF4D4D'
    blue = '#3DE8F7'

    bar_colors = list(map(lambda value: red if value > 0 else blue, values))
    bar_colors = list(np.array(bar_colors)[index_sorted])

    ax.set(xlim=(-100, 100))
    ax.set_xticks([-100, 0, 100])
    ax.set_xticklabels([-100, 0, 100], fontsize=fs)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=fs)
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelleft='off', labelright='on')
    ax.yaxis.set_ticks_position('right')
    ax.set_xlim(-105, 105)

    ax.set_title(title, fontsize=fs)
    ax.barh(yticks, sorted_column_values, align='center', color=bar_colors)

    if show_and_save:
        plt.tight_layout()
        n_concepts = model.parametrizer.nconcept
        save_path = MNIST_IMAGES.joinpath('activation_cpts{}_{}.pdf'.format(n_concepts, label))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()
