import matplotlib.pyplot as plt
import numpy as np
import torch
from api.generator import Generator


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
        plt.title('Generation losses for {}={}, {}={}, method="{}"'.format(r"$\alpha$", p1, r"$\beta$", p2, method))
    else:
        plt.title('Generation losses for {}={}, {}={}'.format(r"$\alpha$", p1, r"$\beta$", p2))
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
        sum_aj = torch.sum(activations ** 2) - ai ** 2
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
                    info = 'M="{}"\n{}={}\n{}={}'.format(method, r"$\alpha$", p1, r"$\beta$", p2)
                else:
                    info = '{}={}\n{}={}'.format(r"$\alpha$", p1, r"$\beta$", p2)
                if prev is None:
                    prev = torch.tensor(generated)
                    prototypes[cpt][extreme][prev] = [activations[cpt], info]
                    losses[cpt][extreme][sample] = [sample_loss, "loss"]
                elif (activations[cpt] > prototypes[cpt][extreme][prev][0] and extreme == "high") or \
                        (activations[cpt] < prototypes[cpt][extreme][prev][0] and extreme == "low"):
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
                    sp = plt.subplot(len(prototypes) * 2, len(prototypes[0]["high"]), i)
                else:
                    sp = plt.subplot(len(prototypes), len(prototypes[0]["high"]) * 2 + 1, i)
                plt.axis("off")
                plt.imshow(sample)
                if show_activations:
                    a = prototypes[cpt][extreme][sample][0]
                    plt.text(2, 33, "a={:.3}".format(a), fontsize=9)
                plt.text(-20, 15, f"Cpt{cpt + 1}\n{extreme}", fontsize=10) if n == 0 else None
                if cpt == 0 and (not compact or (compact and m == 0)):
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


def evaluate(model, dataset, print_freq=1000):
    model.eval()
    correct = 0.
    # dataloader could be faster/more practical
    batch_x = torch.zeros(len(dataset), 1, 28, 28)
    batch_t = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        if print_freq != 0 and i % print_freq == 0:
            print(f"{i}/{len(dataset)}")
        x, t = get_digit(dataset, i)
        batch_x[i, 0, :, :] = x
        batch_t[i] = t
    if print_freq != 0:
        print(f"{len(dataset)}/{len(dataset)}")
        print("Evaluating...")
    with torch.no_grad():
        batch_y = model(batch_x)
        correct = torch.sum(torch.argmax(batch_y, axis=1) == batch_t)
    acc = correct.type(torch.FloatTensor) / len(dataset)
    if print_freq != 0:
        print("accuracy = {:.3}".format(acc))

    return acc


def faithfullness_plot(model, dataset, indx, show_h=False, show_htheta=True):
    # sometimes the model is so certain of a given class that removing a theta_i has no effect
    prob_drop = []
    x, t = get_digit(dataset, indx)
    nconcepts = model.parametrizer.nconcept
    with torch.no_grad():
        theta_x = model.parametrizer(x.view(1, 1, 28, 28))
        h_x = model.conceptizer.encode(x.view(1, 1, 28, 28))
        probs_0 = torch.softmax(model(x.view(1, 1, 28, 28)).squeeze(), dim=0)
        prob_t = probs_0[t]
        for i in range(nconcepts):
            theta_i = torch.tensor(theta_x)
            theta_i[0, i, :] = 0.
            prob_i = torch.softmax(model.aggregator(h_x, theta_i).squeeze(), dim=0)[t]
            prob_drop.append(prob_t - prob_i)
    plt.title("Faithfulness plot for a single sample",fontsize=12)
    plt.xlabel("Concept Index",fontsize=12)

    ax1 = plt.subplot()
    p1 = ax1.bar(range(len(theta_x.squeeze()[:, t])), theta_x.squeeze()[:, t])

    ax1.tick_params(axis='y', colors=p1[0]._facecolor)
    ax1.yaxis.label.set_color(p1[0]._facecolor)
    plt.ylabel(r"Feature Relevance $\theta(x)_i$",fontsize=12)

    ax2 = ax1.twinx()
    p2 = ax2.plot(range(nconcepts), prob_drop, "--", color="darkorange")
    ax2.tick_params(axis='y', colors=p2[0].get_color())
    ax2.yaxis.label.set_color(p2[0].get_color())
    ax2.scatter(range(nconcepts), [float(i) for i in prob_drop], color="darkorange")
    plt.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    plt.ylabel(r"Probability Drop",fontsize=12)
    plt.xticks(range(nconcepts), [str(i) for i in range(1, nconcepts + 1)])
    plt.tight_layout()
    plt.show()

    h_x = h_x.squeeze()
    theta_x = theta_x.squeeze()
    if show_h:
        plt.title("Concept activation",fontsize=12)
        plt.xlabel("Concept Index",fontsize=12)
        plt.ylabel(r"$h(x)_i$",fontsize=12)
        plt.bar(range(nconcepts), h_x)
        plt.xticks(range(nconcepts), [str(i) for i in range(1, nconcepts + 1)])
        plt.show()

    if show_htheta:
        plt.title("Relevance multiplied by concept activation",fontsize=12)
        plt.xlabel("Concept Index",fontsize=12)
        plt.ylabel(r"$\theta(x)_i \cdot h(x)_i$",fontsize=12)
        plt.bar(range(nconcepts), np.array(h_x) * np.array(theta_x[:, t]))
        plt.xticks(range(nconcepts), [str(i) for i in range(1, nconcepts + 1)])
        plt.show()
