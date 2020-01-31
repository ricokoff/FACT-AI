from api.datasets import COMPAS_TEST_SET


def get_sample(index):
    x, t = COMPAS_TEST_SET.__getitem__(index)
    return x, t
