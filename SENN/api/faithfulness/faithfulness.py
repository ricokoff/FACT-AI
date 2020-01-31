import matplotlib.pyplot as plt
import torch

from api.common.mnist import get_digit


def plot_faithfulness(model, index, show_h=False, show_htheta=True):
    # sometimes the model is so certain of a given class that removing a theta_i has no effect
    prob_drop = []
    x, t = get_digit(index)
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
    plt.title("Faithfulness plot for a single sample", fontsize=12)
    plt.xlabel("Concept Index", fontsize=12)

    ax1 = plt.subplot()
    p1 = ax1.bar(range(len(theta_x.squeeze()[:, t])), theta_x.squeeze()[:, t])

    ax1.tick_params(axis='y', colors=p1[0]._facecolor)
    ax1.yaxis.label.set_color(p1[0]._facecolor)
    plt.ylabel(r"Feature Relevance $\theta(x)_i$", fontsize=12)

    ax2 = ax1.twinx()
    p2 = ax2.plot(range(nconcepts), prob_drop, "--", color="darkorange")
    ax2.tick_params(axis='y', colors=p2[0].get_color())
    ax2.yaxis.label.set_color(p2[0].get_color())
    ax2.scatter(range(nconcepts), [float(i) for i in prob_drop], color="darkorange")
    plt.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    plt.ylabel(r"Probability Drop", fontsize=12)
    plt.xticks(range(nconcepts), [str(i) for i in range(1, nconcepts + 1)])
    plt.tight_layout()
    plt.show()

    h_x = h_x.squeeze()
    theta_x = theta_x.squeeze()
    if show_h:
        plt.title("Concept activation", fontsize=12)
        plt.xlabel("Concept Index", fontsize=12)
        plt.ylabel(r"$h(x)_i$", fontsize=12)
        plt.bar(range(nconcepts), h_x)
        plt.xticks(range(nconcepts), [str(i) for i in range(1, nconcepts + 1)])
        plt.show()

    if show_htheta:
        plt.title("Relevance multiplied by concept activation", fontsize=12)
        plt.xlabel("Concept Index", fontsize=12)
        plt.ylabel(r"$\theta(x)_i \cdot h(x)_i$", fontsize=12)
        plt.bar(range(nconcepts), np.array(h_x) * np.array(theta_x[:, t]))
        plt.xticks(range(nconcepts), [str(i) for i in range(1, nconcepts + 1)])
        plt.show()
