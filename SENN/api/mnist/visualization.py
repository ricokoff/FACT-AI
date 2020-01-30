import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

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

def im_act_con_plot(model, index, cuda=False, top_k=6, layout='vertical', return_fig=False, save_path=None):
    data_loader = DataLoader(MNIST_TEST_SET, **{'batch_size': 64, 'num_workers': 9, 'shuffle': False})

    all_norms = []
    num_concepts = model.parametrizer.nconcept
    concept_dim = model.parametrizer.dout

    top_activations = {k: np.array(top_k * [-1000.00]) for k in range(num_concepts)}
    top_examples = {k: top_k * [None] for k in range(num_concepts)}
    all_activs = []
    for idx, (data, target) in enumerate(data_loader):
        # get the inputs
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = torch.tensor(data), torch.tensor(target)
        output = model(data)
        concepts = model.concepts.data
        # pdb.set_trace()
        # concepts[concepts < 0] = 0.0 # This is unncessary if output of H is designed to be > 0.
        if concepts.shape[-1] > 1:
            print('ERROR')
            print(asd.asd)
            activations = np.linalg.norm(concepts, axis=2)
        else:
            activations = concepts

        all_activs.append(activations)
        # if idx == 10:
        #     break

    all_activs = torch.cat(all_activs)
    top_activations, top_idxs = torch.topk(all_activs, top_k, 0)
    top_activations = top_activations.squeeze().t()
    top_idxs = top_idxs.squeeze().t()
    top_examples = {}
    for i in range(num_concepts):
        top_examples[i] = data_loader.dataset.test_data[top_idxs[i]]

    if layout == 'horizontal':
        fig = plt.figure(figsize=(20 + top_k, num_concepts + 1), constrained_layout=True)
        grid = fig.add_gridspec(num_concepts, 10 + top_k, wspace=0.4, hspace=0.3)
        ax1 = fig.add_subplot(grid[:, 0:4])
        ax2 = fig.add_subplot(grid[:, 5:9])
        fontsize = 18
    else:
        fig = plt.figure(figsize=(num_concepts + 1, 20 + top_k), constrained_layout=True)
        grid = fig.add_gridspec(10 + top_k, num_concepts, wspace=0.4, hspace=0.3)
        ax1 = fig.add_subplot(grid[0:4, :])
        ax2 = fig.add_subplot(grid[5:9, :])
        fontsize = 14

    digitim, _ = get_digit(MNIST_TEST_SET, index)
    plot_digit(digitim.squeeze(), "test", title='Number ' + str(index), ax=ax1, fs=fontsize)
    plot_activation(model, digitim, "test", title='SENN', ax=ax2, fs=fontsize)

    for i in range(num_concepts):
        for j in range(top_k):
            if layout == 'horizontal':
                ax = fig.add_subplot(grid[i, j + 10])
            else:
                ax = fig.add_subplot(grid[j + 10, i])

            l = i * top_k + j
            # print(i,j)
            # print(top_examples[i][j].shape)
            ax.imshow(top_examples[i][j], cmap='Greys', interpolation='nearest')
            if layout == 'vertical':
                ax.axis('off')
                if j == 0:
                    ax.set_title('Cpt {}'.format(i + 1), fontsize=fontsize)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.set_xticks([])
                for side in ['top', 'right', 'bottom', 'left']:
                    ax.spines[side].set_visible(False)
                if i == 0:
                    ax.set_title('Pr{}'.format(j + 1), fontsize=fontsize)
                if j == 0:
                    ax.set_ylabel('Ct{}'.format(i + 1), fontsize=fontsize)

    if layout == 'vertical':
        fig.subplots_adjust(wspace=0.01, hspace=0.1)
    else:
        fig.subplots_adjust(wspace=0.1, hspace=0.01)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()

def plot_accuracy(accuracy):
    lambdas = accuracy.keys()
    values = accuracy.values()

