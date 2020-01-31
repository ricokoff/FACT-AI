import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader

from api.common.mnist import get_digit, plot_activation, plot_digit
from api.datasets import MNIST_TEST_SET
from api.folders import MNIST_IMAGES


def plot_digit_activation_concept_grid(model, index, top_k=6, layout='vertical'):
    data_loader = DataLoader(MNIST_TEST_SET, **{'batch_size': 64, 'num_workers': 9, 'shuffle': False})

    num_concepts = model.parametrizer.nconcept

    all_activs = []
    for idx, (data, target) in enumerate(data_loader):
        # get the inputs
        data, target = torch.tensor(data), torch.tensor(target)
        output = model(data)
        concepts = model.concepts.data
        activations = concepts
        all_activs.append(activations)

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

    digitim, _ = get_digit(index)
    plot_digit(digitim.squeeze(), "test", title='Index ' + str(index), ax=ax1, fs=fontsize)
    plot_activation(model, digitim, "test", title='SENN', ax=ax2, fs=fontsize)

    for i in range(num_concepts):
        for j in range(top_k):
            if layout == 'horizontal':
                ax = fig.add_subplot(grid[i, j + 10])
            else:
                ax = fig.add_subplot(grid[j + 10, i])

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

    save_path = MNIST_IMAGES.joinpath('digit_activation_concept_grid.pdf')
    plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
