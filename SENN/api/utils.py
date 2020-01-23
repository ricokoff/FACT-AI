import os

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import MNIST

path = Path(__file__)

PROJECT_NAME = 'SENN'

while path.name != PROJECT_NAME:
    path = path.parent

MODELS_FOLDER = path.joinpath('models')
DATA_FOLDER = path.joinpath('data')
IMAGES_FOLDER = path.joinpath('images')

for folder in [MODELS_FOLDER, DATA_FOLDER, IMAGES_FOLDER.joinpath('mnist'), IMAGES_FOLDER.joinpath('compas')]:
    if not os.path.isdir(folder):
        os.makedirs(folder)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

MNIST_TRAIN_SET = MNIST(DATA_FOLDER.joinpath('MNIST'), train=True, download=False, transform=transform)
MNIST_TEST_SET = MNIST(DATA_FOLDER.joinpath('MNIST'), train=False, download=False, transform=transform)
