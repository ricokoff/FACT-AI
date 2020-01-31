import os
from pathlib import Path

PATH = Path(__file__)

PROJECT_NAME = 'FACT-AI'

while PATH.name != PROJECT_NAME:
    PATH = PATH.parent

MNIST = 'mnist'
COMPAS = 'compas'

MODELS_FOLDER = PATH.joinpath('models')
DATA_FOLDER = PATH.joinpath('data')
IMAGES_FOLDER = PATH.joinpath('images')

MNIST_MODELS = MODELS_FOLDER.joinpath(MNIST)
COMPAS_MODELS = MODELS_FOLDER.joinpath(COMPAS)

MNIST_DATA = DATA_FOLDER.joinpath(MNIST)
COMPAS_DATA = DATA_FOLDER.joinpath(COMPAS)

MNIST_IMAGES = IMAGES_FOLDER.joinpath(MNIST)
COMPAS_IMAGES = IMAGES_FOLDER.joinpath(COMPAS)

for folder in [MNIST_IMAGES, COMPAS_IMAGES]:
    if not os.path.isdir(folder):
        os.makedirs(folder)
