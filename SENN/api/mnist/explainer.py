import torch
from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from torch.autograd import Variable

from api.load import load_mnist
from api.refactored_mnist import get_tensor


def explain(model, x):
    methods = ['saliency', 'grad*input', 'intgrad', 'elrp', 'deeplift', 'occlusion', 'shapley_sampling']
    X = x
    with torch.no_grad():
        prediction = model(X)
    T = model.thetas.data.numpy().squeeze()
    with DeepExplain(session=K.get_session()) as de:
        for m in methods:
            attributions = de.explain(m, T, X, x.reshape(1, *x.shape))
            print(attributions)

model = load_mnist()
x = get_tensor(5)

explain(model, x)