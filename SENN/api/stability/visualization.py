import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from api.datasets import COMPAS_FEAT_NAMES
from api.folders import COMPAS_IMAGES, MNIST_IMAGES
from api.common.mnist import get_digit, plot_activation, plot_digit


def gaussian_perturbation(x):
    with torch.no_grad():
        noise = torch.tensor(.5 * torch.randn(x.size()))
        return x + noise


def plot_digit_activation(model, x, label, layout='horizontal', ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        nrows = 1 if layout == 'horizontal' else 2
        ncols = 2 if layout == 'horizontal' else 1
        _, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    plot_digit(x, label, ax=ax[0])
    plot_activation(model, x, label, '', ax=ax[1])

    if show_and_save:
        plt.tight_layout()
        n_concepts = model.parametrizer.nconcept
        save_path = MNIST_IMAGES.joinpath('digit_{}_activation_cpts{}.pdf'.format(label, n_concepts))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_digit_noise(digit, number_of_samples=0, ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        nrows = 1
        ncols = number_of_samples + 1
        _, ax = plt.subplots(nrows, ncols)

    original, _ = get_digit(digit)
    plot_digit(original, digit, ax=ax[0])
    samples = [gaussian_perturbation(original) for _ in range(number_of_samples)] if number_of_samples > 0 else []
    for i, x in enumerate(samples):
        plot_digit(x, digit, 'Perturbation {}'.format(i + 1), ax[i + 1])

    if show_and_save:
        plt.tight_layout()
        save_path = MNIST_IMAGES.joinpath('digit_{}_noise.pdf'.format(digit))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_digit_noise_activation_regularized(model, digit, number_of_samples, ax=None):
    num_concepts = model.parametrizer.nconcept
    show_and_save = False
    if ax is None:
        show_and_save = True
        nrows = 2
        ncols = number_of_samples + 1
        _, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * num_concepts / 5. * 2.))

    original, _ = get_digit(digit)
    plot_digit(original, digit, ax=ax[0, 0])
    samples = [gaussian_perturbation(original) for _ in range(number_of_samples)] if number_of_samples > 0 else []

    for i, x in enumerate(samples):
        plot_digit(x, digit, 'Perturbation {}'.format(i + 1), ax[0, i + 1])

    for i, x in enumerate([original] + samples):
        plot_activation(model, x, digit, '', ax[1, i])

    if show_and_save:
        plt.tight_layout()
        save_path = MNIST_IMAGES.joinpath('digit_{}_noise.pdf'.format(digit))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_digit_noise_activation_regularized_unregularized(model, unregularized_model, digit, number_of_samples,
                                                          ax=None, fs=10):
    num_concepts = model.parametrizer.nconcept
    show_and_save = False
    if ax is None:
        show_and_save = True
        nrows = 3
        ncols = number_of_samples + 1
        _, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * num_concepts / 5. * 2.))

    original, _ = get_digit(digit)
    plot_digit(original, digit, ax=ax[0, 0], fs=fs)
    samples = [gaussian_perturbation(original) for _ in range(number_of_samples)] if number_of_samples > 0 else []

    for i, x in enumerate(samples):
        plot_digit(x, digit, 'Perturbation {}'.format(i + 1), ax[0, i + 1], fs)

    for i, x in enumerate([original] + samples):
        plot_activation(unregularized_model, x, digit, 'Unregularized', ax[1, i], fs)

    for i, x in enumerate([original] + samples):
        plot_activation(model, x, digit, 'Regularized', ax[2, i], fs)

    if show_and_save:
        plt.tight_layout()
        save_path = MNIST_IMAGES.joinpath('digit_{}_noise_activation_unregularized.pdf'.format(digit))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


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
        save_path = COMPAS_IMAGES.joinpath('input_values.pdf'.format())
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
    features = list(parameters.keys())[::-1]
    values = list(parameters.values())[::-1]
    maximum_value = np.absolute(np.array(values)).max()
    values = ((np.array(values) / maximum_value) * 100)

    yticks = np.arange(len(features)) + 0.5
    yticklabels = list(np.array(features))

    red = '#FF4D4D'
    blue = '#3DE8F7'

    bar_colors = list(map(lambda value: blue if value > 0 else red, values))

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
    ax.barh(yticks, values, align='center', color=bar_colors)

    if show_and_save:
        plt.tight_layout()
        save_path = COMPAS_IMAGES.joinpath('explanations.pdf'.format())
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_input_values_regularized_unregularized_explanation(model, unregularized_model, xs, ax=None):
    show_and_save = False
    nrows = len(xs)
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))

    for i, x in enumerate(xs):
        plot_input_values(model, x, ax=ax[i, 0] if nrows > 1 else ax[0])
        plot_explanation(model, x, title=r'Relevance Score $\theta$(x) (Scaled)',
                         ax=ax[i, 1] if nrows > 1 else ax[1])
        plot_explanation(unregularized_model, x, title=r'Relevance Score $\theta$(x) (Scaled)', show_labels=False,
                         ax=ax[i, 2] if nrows > 1 else ax[2])

    if show_and_save:
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)
        save_path = COMPAS_IMAGES.joinpath('input_values_regularized_unregularized_explanation.pdf')
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()
