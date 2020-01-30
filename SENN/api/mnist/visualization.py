import matplotlib.pyplot as plt
import numpy as np
import torch

from api.utils import MNIST_TEST_SET, IMAGES_FOLDER


# def get_tensor(digit):
#     indices = [(MNIST_TEST_SET.targets == i).nonzero().reshape(1, -1).view(-1) for i in range(10)]
#     index = np.random.choice(indices[digit])
#     example = MNIST_TEST_SET[index][0].view(1, 1, 28, 28)
#     return example
#
from test_utils import get_digit

def gaussian_perturbation(x):
    with torch.no_grad():
        noise = torch.tensor(.5 * torch.randn(x.size()))
        return x + noise


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
    ax.set_yticks([y_size - 1, 0])
    ax.set_yticklabels([0, y_size], fontsize=fs)

    ax.set_title(title, fontsize=fs)
    ax.title.set_position([.5, 1.05])
    ax.imshow(x)

    if show_and_save:
        plt.tight_layout()
        plt.subplots_adjust(bottom=0)
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}.pdf'.format(label))
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
    ax.set_xlim(-105, 105)

    ax.set_title(title, fontsize=fs)
    ax.barh(yticks, sorted_column_values, align='center', color=bar_colors)

    if show_and_save:
        plt.tight_layout()
        n_concepts = model.parametrizer.nconcept
        save_path = IMAGES_FOLDER.joinpath('mnist', 'activation_cpts{}_{}.pdf'.format(n_concepts, label))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


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
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}_activation_cpts{}.pdf'.format(label, n_concepts))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_digit_noise(digit, number_of_samples=0, ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        nrows = 1
        ncols = number_of_samples + 1
        _, ax = plt.subplots(nrows, ncols)

    original, _ = get_digit(MNIST_TEST_SET, digit)
    plot_digit(original, digit, ax=ax[0])
    samples = [gaussian_perturbation(original) for _ in range(number_of_samples)] if number_of_samples > 0 else []
    for i, x in enumerate(samples):
        plot_digit(x, digit, 'Perturbation {}'.format(i + 1), ax[i + 1])

    if show_and_save:
        plt.tight_layout()
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}_noise.pdf'.format(digit))
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

    original, _ = get_digit(MNIST_TEST_SET, digit)
    plot_digit(original, digit, ax=ax[0, 0])
    samples = [gaussian_perturbation(original) for _ in range(number_of_samples)] if number_of_samples > 0 else []

    for i, x in enumerate(samples):
        plot_digit(x, digit, 'Perturbation {}'.format(i + 1), ax[0, i + 1])

    for i, x in enumerate([original] + samples):
        plot_activation(model, x, digit, '', ax[1, i])

    if show_and_save:
        plt.tight_layout()
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}_noise.pdf'.format(digit))
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

    original, _ = get_digit(MNIST_TEST_SET, digit)
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
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}_noise_activation_unregularized.pdf'.format(digit))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()
