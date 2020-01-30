import torch

from api.utils import RegLambda, HType, NConcepts, MODELS_FOLDER

def load_compas(reg_lambda=RegLambda.E4, show_specs=True):
    path = 'grad3_Hinput_Thsimple_Reg{:0.0e}_LR0.0002'.format(reg_lambda.value)
    if show_specs:
        print('Loading MNIST model ' + path + ':')
        show_model_specs(reg_lambda)
    best = torch.load(MODELS_FOLDER.joinpath('compas', path, 'model_best.pth.tar'), map_location=torch.device('cpu'))
    model = best['model']
    model.eval()
    return model


def load_mnist(reg_lambda=RegLambda.E4, h_type=HType.CNN, n_concepts=NConcepts.FIVE, show_specs=True):
    path = ''
    if h_type == HType.CNN:
        path = 'grad3_Hcnn_Thsimple_Cpts{}_Reg{:0.0e}_Sp2e-05_LR0.0002'.format(n_concepts.value, reg_lambda.value)
    elif h_type == HType.INPUT:
        path = 'grad3_Hinput_Thsimple_Reg{:0.0e}_LR0.0002'.format(reg_lambda.value)
    if show_specs:
        print('Loading MNIST model ' + path + ':')
        show_model_specs(reg_lambda, h_type, n_concepts)
    best = torch.load(MODELS_FOLDER.joinpath('mnist', path, 'model_best.pth.tar'), map_location=torch.device('cpu'))
    model = best['model']
    model.eval()
    return model


def show_model_specs(reg_lambda, h_type=HType.INPUT, n_concepts=None):
    reg_format = '{:0.0e}' if reg_lambda.value != 0 and reg_lambda.value != 1 else '{}'
    print(f'> conceptizer type        = {h_type.name.lower()}')
    print(f'> number of concepts      = {n_concepts.value}') if not h_type == HType.INPUT else ''
    print('> sparsity parameter      = 2e-05') if not h_type == HType.INPUT else ''
    print(('> regularization strength = ' + reg_format).format(reg_lambda.value))
