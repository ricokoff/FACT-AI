import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from api.generator import Generator

import torch.utils.data.dataloader as dataloader

from api.utils import MNIST_TEST_SET, MNIST_TRAIN_SET, IMAGES_FOLDER


def get_digit_image(digit):
    indices = [(MNIST_TEST_SET.targets == i).nonzero().reshape(1, -1).view(-1) for i in range(10)]
    index = np.random.choice(indices[digit])
    example = MNIST_TEST_SET[index][0].view(1, 1, 28, 28)
    return example


def gaussian_perturbation(x):
    with torch.no_grad():
        noise = Variable(.5 * torch.randn(x.size()))
        return x + noise


def plot_digit(x, label, title='Original', ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots()

    x_size, y_size = x.size()

    ax.set(xlim=(0, x_size - 1), ylim=(0, y_size - 1))
    ax.set_xticks([0, x_size - 1])
    ax.set_xticklabels([0, x_size])
    ax.set_yticks([y_size - 1, 0])
    ax.set_yticklabels([0, y_size])

    ax.set_title(title)
    ax.imshow(x)

    if show_and_save:
        plt.tight_layout()
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}.pdf'.format(label))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_activation(model, x, label, title='SENN', ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        _, ax = plt.subplots()

    model.eval()
    with torch.no_grad():
        prediction = model(x).max(1)[1]
        theta = model.thetas.data.numpy().squeeze()
    concept_names = ['C{}'.format(i) for i in range(theta.shape[0])]
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
    ax.set_xticklabels([-100, 0, 100])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(-105, 105)

    ax.set_title(title)
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

    plot_digit(x.squeeze(), label, ax=ax[0])
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
        _, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    original = get_digit_image(digit)
    plot_digit(original.squeeze(), digit, ax=ax[0])
    samples = [gaussian_perturbation(original) for _ in range(number_of_samples)] if number_of_samples > 0 else []
    for i, x in enumerate(samples):
        plot_digit(x.squeeze(), digit, 'Perturbation {}'.format(i + 1), ax[i + 1])

    if show_and_save:
        plt.tight_layout()
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}_noise.pdf'.format(digit))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def plot_digit_noise_activation(model, digit, number_of_samples, ax=None):
    num_concepts = model.parametrizer.nconcept
    show_and_save = False
    if ax is None:
        show_and_save = True
        nrows = 2
        ncols = number_of_samples + 1
        _, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * num_concepts / 5. * 2.))

    original = get_digit_image(digit)
    plot_digit(original.squeeze(), digit, ax=ax[0, 0])
    samples = [gaussian_perturbation(original) for _ in range(number_of_samples)] if number_of_samples > 0 else []

    for i, x in enumerate(samples):
        plot_digit(x.squeeze(), digit, 'Perturbation {}'.format(i + 1), ax[0, i + 1])

    for i, x in enumerate([original] + samples):
        plot_activation(model, x, digit, '', ax[1, i])

    if show_and_save:
        plt.tight_layout()
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}_noise.pdf'.format(digit))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


# TODO: REFACTOR
def _concept_grid(model, data_loader, top_k=6, layout='vertical', fig=None):
    num_concepts = model.parametrizer.nconcept
    all_activs = []
    for i in range(len(data_loader)):
        concepts = model.concepts.data
        if concepts.shape[-1] > 1:
            print('ERROR')
            activations = np.linalg.norm(concepts, axis=2)
        else:
            activations = concepts
        all_activs.append(activations)

    all_activs = torch.cat(all_activs)
    _, top_idxs = torch.topk(all_activs, top_k, 0)
    top_idxs = top_idxs.squeeze().t()
    top_examples = {}
    for i in range(num_concepts):
        top_examples[i] = data_loader.dataset.test_data[top_idxs[i]]

    if layout == 'horizontal':
        num_cols = top_k
        num_rows = num_concepts
        figsize = (num_cols, 1.2 * num_rows)
    else:
        num_cols = num_concepts
        num_rows = top_k
        figsize = (1.4 * num_cols, num_rows)

    axes = fig.add_subplot(2, 2, 3)
    # fig, axes = ax.subplots(figsize=figsize, nrows=num_rows, ncols=num_cols)

    for i in range(num_concepts):
        for j in range(top_k):
            pos = (i, j) if layout == 'horizontal' else (j, i)

            l = i * top_k + j
            axes[pos].imshow(top_examples[i][j], cmap='Greys', interpolation='nearest')
            if layout == 'vertical':
                axes[pos].axis('off')
                if j == 0:
                    axes[pos].set_title('Cpt {}'.format(i + 1), fontsize=24)
            else:
                axes[pos].set_xticklabels([])
                axes[pos].set_yticklabels([])
                axes[pos].set_yticks([])
                axes[pos].set_xticks([])
                for side in ['top', 'right', 'bottom', 'left']:
                    axes[i, j].spines[side].set_visible(False)
                if i == 0:
                    axes[pos].set_title('Proto {}'.format(j + 1))
                if j == 0:
                    axes[pos].set_ylabel('Concept {}'.format(i + 1), rotation=90)

    cols = ['Prot.{}'.format(col) for col in range(1, num_cols + 1)]
    rows = ['Concept # {}'.format(row) for row in range(1, num_rows + 1)]

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    plt.tight_layout()

    if layout == 'vertical':
        fig.subplots_adjust(wspace=0.01, hspace=0.1)
    else:
        fig.subplots_adjust(wspace=0.1, hspace=0.01)

    plt.show()
    return fig, ax


# TODO: REFACTOR
def concept_grid(model, fig=None):
    test_loader = dataloader.DataLoader(MNIST_TEST_SET, **{'batch_size': 64, 'num_workers': 9, 'shuffle': False})
    return _concept_grid(model, test_loader, top_k=10, fig=fig)


# TODO: REFACTOR
def plot_digit_activation_grid(model, x, label, ax=None):
    show_and_save = False
    if ax is None:
        show_and_save = True
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), subplot_kw={'xticks': [], 'yticks': []})

    plot_digit(x.squeeze(), label, ax=axs[0])
    plot_activation(model, x, label, ax=axs[1])

    if show_and_save:
        plt.tight_layout()
        n_concepts = model.parametrizer.nconcept
        save_path = IMAGES_FOLDER.joinpath('mnist', 'digit_{}_activation_grid_cpts{}.pdf'.format(label, n_concepts))
        plt.savefig(str(save_path), bbox_inches='tight', format='pdf', dpi=300)
        plt.show()


def get_digit(dataset, indx):
    d, t = dataset.__getitem__(indx)
    return d.view(28, 28), t


# find lowest (find="low") or highest (find="high") value in dict
def dic_find(dic, find):
    best = 1e9 if find == "low" else -1e9
    for key in dic:
        temp = dic[key][0]
        if (temp < best and find == "low") or (temp > best and find == "high"):
            bestkey = key
            best = temp
    return bestkey


def find_prototypes(model, dataset, prototypes, Nsamples=1, print_freq=5000):
    print("Checking dataset for best prototypes...") if print_freq != 0 else ""
    for i in range(len(dataset)):
        if print_freq != 0 and i % print_freq == 0:
            print(f"{i}/{len(dataset)}")
        digit, _ = get_digit(dataset, i)
        encoded = model.conceptizer.encode(digit.view(1, 1, 28, 28)).squeeze()
        # check for every concept if this sample is better than prev ones
        for cpt in prototypes.keys():
            lowkey_of_high = dic_find(prototypes[cpt]["high"], "low")
            highkey_of_low = dic_find(prototypes[cpt]["low"], "high")
            # if the sample is better, save its index in the dataset (for speed)
            if encoded[cpt] > prototypes[cpt]["high"][lowkey_of_high][0]:
                if len(prototypes[cpt]["high"]) == Nsamples:
                    prototypes[cpt]["high"].pop(lowkey_of_high)
                prototypes[cpt]["high"][i] = [encoded[cpt], "data"]
            if encoded[cpt] < prototypes[cpt]["low"][highkey_of_low][0]:
                if len(prototypes[cpt]["low"]) == Nsamples:
                    prototypes[cpt]["low"].pop(highkey_of_low)
                prototypes[cpt]["low"][i] = [encoded[cpt], "data"]
    print(f"{len(dataset)}/{len(dataset)}") if print_freq != 0 else ""
    # convert the indices of the images to the images themselves
    for cpt in prototypes.keys():
        for extreme in prototypes[cpt].keys():
            for indx in dict(prototypes[cpt][extreme]).keys():
                prototypes[cpt][extreme][get_digit(dataset, indx)[0]] = prototypes[cpt][extreme][indx]
                prototypes[cpt][extreme].pop(indx)
    return prototypes


def show_losses(losses, p1, p2, method):
    if p1 != 0:
        plt.title('Generation losses for {}={}, {}={}, method="{}"'.format(r"$\alpha$",p1,r"$\beta$",p2,method))
    else:
        plt.title('Generation losses for {}={}, {}={}'.format(r"$\alpha$",p1,r"$\beta$",p2))
    for extreme in losses[0].keys():
        linestyle = "-" if extreme == "high" else "--"
        for cpt in losses.keys():
            #             for sample in losses[cpt][extreme].keys(): #plot one for better view
            # be aware that these are just the first samples tried, not the ones of the best samples
            plt.plot(losses[cpt][extreme][0][0], linestyle, label=f"C{cpt + 1} {extreme}")
    plt.legend(ncol=2, bbox_to_anchor=(1, 1))
    plt.show()


# for p(x)~N(0,1), see:
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
def KL(q):
    return -torch.log(torch.std(q)) + (torch.std(q) ** 2 + torch.mean(q) ** 2) / 2 - 0.5


def criterion(activations, cpt, generated, sign, p1=1, p2=1, method="zero"):
    ai = activations[cpt]
    if method == "zero":  # force all other activations to zero
        sum_aj = torch.sum(activations**2) - ai**2
        return sign * ai + p1 * 0.5 * sum_aj + p2 * KL(generated)
    elif method == "diff":  # force largest difference with other activations
        sum_aj = torch.sum(activations) - ai
        return sign * ai - p1 * sign * sum_aj + p2 * KL(generated)


def generate_prototypes(model, prototypes, Nsteps=100, Nsamples=1, lr=0.1, p1=1, p2=1, print_freq=1, x0=None,
                        show_loss=False, method="zero"):
    print("Generating prototypes...") if print_freq != 0 else None
    for param in model.parameters():
        param.requires_grad = False

    losses = empty_prototypes(model)
    generator = Generator(model.conceptizer)
    for cpt in prototypes.keys():
        if print_freq != 0 and cpt % print_freq == 0:
            print(f"{cpt}/{len(prototypes.keys())}")
        for extreme in prototypes[cpt].keys():
            sign = -1. if extreme == "high" else 1.
            prev = None
            for sample in range(Nsamples):
                sample_loss = []
                generator.initialize(x0=x0)  # reset generator
                optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
                for step in range(Nsteps):
                    optimizer.zero_grad()
                    activations = generator.forward()
                    loss = criterion(activations, cpt, generator.generated, sign, p1=p1, p2=p2, method=method)
                    loss.backward()
                    optimizer.step()
                    sample_loss.append(loss)
                generated = torch.tensor(generator.generated.detach())
                if p1 != 0:
                    info = 'M="{}"\n{}={}\n{}={}'.format(method,r"$\alpha$",p1,r"$\beta$",p2)
                else:
                    info = '{}={}\n{}={}'.format(r"$\alpha$",p1,r"$\beta$",p2)
                if prev is None:
                    prev = torch.tensor(generated)
                    prototypes[cpt][extreme][prev] = [activations[cpt], info]
                    losses[cpt][extreme][sample] = [sample_loss, "loss"]
                elif (activations[cpt] > prototypes[cpt][extreme][prev][0] and extreme=="high") or \
                     (activations[cpt] < prototypes[cpt][extreme][prev][0] and extreme=="low"):
                    prototypes[cpt][extreme].pop(prev)
                    prev = torch.tensor(generated)
                    prototypes[cpt][extreme][prev] = [activations[cpt], info]
                    losses[cpt][extreme][sample] = [sample_loss, "loss"]
    print(f"{len(prototypes.keys())}/{len(prototypes.keys())}") if print_freq != 0 else ""
    if show_loss:
        show_losses(losses, p1, p2, method)
    return prototypes


def empty_prototypes(model, dummies=False):
    nconcepts = model.conceptizer.conv2.out_channels
    if dummies:
        return {i: {"high": {-1: [-1e9, "dummy"]}, "low": {-1: [1e9, "dummy"]}} for i in range(nconcepts)}
    else:
        return {i: {"high": {}, "low": {}} for i in range(nconcepts)}


def visualize_cpts(model, dataset, p1=[1], p2=[1],
                   method=["zero"],
                   x0=None,
                   show_loss=False,
                   print_freqs=[5000, 2],
                   show_activations=False,
                   return_prototypes=False,
                   best_of=1,
                   compact=False):
    # visualize concepts with images that max- or minimize their activations
    # (i): find the best samples from dataset, i.e. the prototypes
    # (ii): generate samples using gradient descent
    # (iii): visualize results
    # p1, p2 and method are corresponding lists specifying param settings for (ii)
    # x0 is the initial guess for the generator: random if not given
    # print_freqs specifies print frequencies for (i) and (ii) respectively
    # best_of specifies how many times the generator will try to get a good activation
    # TO DO: input checks

    # example of use:
    # prototypes = utils.visualize_cpts(model, test,
    #                              p1=[0,0,1,1],
    #                              p2=[0,10,0,10],
    #                              method=["zero"]*4,
    #                              x0=None,
    #                              print_freqs=[1000,2],
    #                              show_loss=False,
    #                              show_activations=True,
    #                              return_prototypes=True,
    #                              best_of=2,
    #                              conpact=True)

    prototypes = empty_prototypes(model, dummies=True)
    prototypes = find_prototypes(model, dataset, prototypes,
                                 print_freq=print_freqs[0])

    p1 = [p1] if not isinstance(p1, list) else p1
    p2 = [p2] if not isinstance(p2, list) else p2
    method = [method] if not isinstance(method, list) else method

    for setting in range(len(p1)):
        prototypes = generate_prototypes(model, prototypes,
                                         p1=p1[setting],
                                         p2=p2[setting],
                                         method=method[setting],
                                         x0=x0,
                                         show_loss=show_loss,
                                         print_freq=print_freqs[1],
                                         Nsamples=best_of)

    print("Setting up visualizations...")
    i = 1
    if compact:
        plt.figure(figsize=(2 + len(p1), 1 + 2 * len(prototypes)))
    else:
        plt.figure(figsize=(4 + len(p1) * 2, 1 + len(prototypes)))
    for cpt in prototypes.keys():
        for m, extreme in enumerate(prototypes[cpt].keys()):
            for n, sample in enumerate(prototypes[cpt][extreme].keys()):
                if compact:
                    sp = plt.subplot(len(prototypes)*2, len(prototypes[0]["high"]), i)
                else:
                    sp = plt.subplot(len(prototypes), len(prototypes[0]["high"]) * 2 + 1, i)
                plt.axis("off")
                plt.imshow(sample)
                if show_activations:
                    a = prototypes[cpt][extreme][sample][0]
                    plt.text(2, 33, "a={:.3}".format(a), fontsize=9)
                plt.text(-20, 15, f"Cpt{cpt + 1}\n{extreme}", fontsize=10) if n == 0 else None
                if cpt == 0 and (not compact or (compact and m==0)):
                    info = prototypes[cpt][extreme][sample][1]
                    if info == "data":
                        plt.title("Prototype\nfrom data", fontsize=9)
                    else:
                        plt.title(f"Generated\n{info}", fontsize=9)
                i += 1
            i += 1 if m == 0 and not compact else 0
    if show_activations:
        plt.subplots_adjust(hspace=.23, wspace=0.1)
    else:
        plt.subplots_adjust(hspace=.1, wspace=0.1)
    plt.show()
    if return_prototypes:
        return prototypes
    else:
        return

def evaluate(model, dataset, print_freq=1000, return_acc=False):
    model.eval()
    correct = 0.
    #dataloader could be faster/more practical
    batch_x = torch.zeros(len(dataset),1,28,28)
    batch_t = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        if print_freq != 0 and i % print_freq == 0:
            print(f"{i}/{len(dataset)}")
        x, t = get_digit(dataset, i)
        batch_x[i,0,:,:] = x
        batch_t[i] = t
    if print_freq != 0:
        print(f"{len(dataset)}/{len(dataset)}")
        print("Evaluating...")
    with torch.no_grad():
        batch_y = model(batch_x)
        correct = torch.sum(torch.argmax(batch_y, axis=1) == batch_t)
    acc = correct.type(torch.FloatTensor)/len(dataset)
    if print_freq != 0:
        print("accuracy = {:.3}".format(acc))
    if return_acc:
        return acc
    return


def faith_evaluate(model, nconcept, dataset, print_freq=1000, return_acc=False, cuda=False):
    model.eval()
    if cuda:
        model.cuda()
    accuracies = {}
    for con in range(nconcept):
        correct = 0.
        #dataloader could be faster/more practical
        batch_x = torch.zeros(len(dataset),1,28,28)
        batch_t = torch.zeros(len(dataset))
        for i in range(len(dataset)):
            if print_freq != 0 and i % print_freq == 0:
                print(f"{i}/{len(dataset)}")
            x, t = get_digit(dataset, i)
            batch_x[i,0,:,:] = x
            batch_t[i] = t
        if print_freq != 0:
            print(f"{len(dataset)}/{len(dataset)}")
            print("Evaluating...")
        with torch.no_grad():
            if cuda:
                batch_x, batch_t = batch_x.cuda(), batch_t.cuda()
            h_x, _ = model.conceptizer(batch_x)
            h_x[:,con,:] = 0
            thetas = model.parametrizer(batch_x)
            batch_y = model.aggregator(h_x, thetas)
            correct = torch.sum(torch.argmax(batch_y, axis=1) == batch_t)
        acc = correct.type(torch.FloatTensor)/len(dataset)
        if print_freq != 0:
            print("accuracy = {:.3}".format(acc))
        if return_acc:
            accuracies[con] = acc

    return accuracies
