

from typing import Literal


AVAILABLE_DATASETS = ['iris', 'mnist', 'cifar10', 'wildcam']
DataMode = Literal['iris', 'mnist', 'cifar10', 'wildcam']


AVAILABLE_MODELS = ['fc', 'lenet5', 'vgg11', 'wideresnet50']
ModelMode = Literal['fc', 'lenet5', 'vgg11', 'wideresnet50']   