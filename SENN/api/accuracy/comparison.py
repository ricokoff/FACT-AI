import matplotlib.pyplot as plt
import numpy as np

from api.parameters import RegLambda


def plot_accuracy_comparison(accuracies, titles):
    x = [lmbd.value for lmbd in RegLambda]
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
            ax[i, j].set_xticklabels([('{:0.0e}' if i not in [0, 1] else '{}').format(i) for i in x])
            ax[i, j].plot(xticks, y, 'go', alpha=0.9, color=colors[i + j * 2])
            ax[i, j].plot(xticks, y, '--', alpha=0.9, color=colors[i + j * 2 + 1])
            ax[i, j].set_title(titles[i + j * 2])
            ax[i, j].set_xlabel(r"$\lambda$")
            ax[i, j].set_ylabel("Accuracy")
            for n, m in zip(xticks, y):
                ax[i, j].text(n, m, "%.2f" % round(float(m), 2), va="bottom")
    plt.tight_layout()
    plt.show()
