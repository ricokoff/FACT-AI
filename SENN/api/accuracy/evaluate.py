import torch

from api.datasets import COMPAS_TEST_SET, MNIST_TEST_SET
from api.common.compas import get_sample
from api.common.mnist import get_digit


def evaluate_compas(model, print_freq=0):
    batch_x = torch.zeros(len(COMPAS_TEST_SET), len(get_sample(0)[0]))
    batch_t = torch.zeros(len(COMPAS_TEST_SET))
    for i in range(len(COMPAS_TEST_SET)):
        if print_freq != 0 and i % print_freq == 0:
            print(f"{i}/{len(COMPAS_TEST_SET)}")
        x, t = get_sample(i)
        batch_x[i, :] = x
        batch_t[i] = t
    if print_freq != 0:
        print(f"{len(COMPAS_TEST_SET)}/{len(COMPAS_TEST_SET)}")
        print("Evaluating...")
    with torch.no_grad():
        batch_y = model(batch_x).squeeze()
        correct = torch.sum(torch.round(batch_y) == batch_t)
    acc = correct.type(torch.FloatTensor) / len(COMPAS_TEST_SET)
    if print_freq != 0:
        print("accuracy = {:.3}".format(acc))

    return acc


def evaluate_mnist(model, print_freq=0):
    batch_x = torch.zeros(len(MNIST_TEST_SET), 1, 28, 28)
    batch_t = torch.zeros(len(MNIST_TEST_SET))
    for i in range(len(MNIST_TEST_SET)):
        if print_freq != 0 and i % print_freq == 0:
            print(f"{i}/{len(MNIST_TEST_SET)}")
        x, t = get_digit(i)
        batch_x[i, 0, :, :] = x
        batch_t[i] = t
    if print_freq != 0:
        print(f"{len(MNIST_TEST_SET)}/{len(MNIST_TEST_SET)}")
        print("Evaluating...")
    with torch.no_grad():
        batch_y = model(batch_x)
        correct = torch.sum(torch.argmax(batch_y, axis=1) == batch_t)
    acc = correct.type(torch.FloatTensor) / len(MNIST_TEST_SET)
    if print_freq != 0:
        print("accuracy = {:.3}".format(acc))

    return acc
