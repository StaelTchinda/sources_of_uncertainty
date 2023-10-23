

from typing import Literal
from network.pruning import pruner as pruning_wrapper

AVAILABLE_DATASETS = ['iris', 'mnist', 'not_mnist', 'cifar10', 'wildcam']
DataMode = Literal['iris', 'mnist', 'not_mnist', 'cifar10', 'wildcam']


AVAILABLE_MODELS = ['fc', 'lenet5', 'vgg11', 'vgg13', 'vgg16', 'wideresnet50']
ModelMode = Literal['fc', 'lenet5', 'vgg11', 'vgg13', 'vgg16', 'wideresnet50']   

AVAILABLE_BAYESIAN_MODES = ['laplace', 'mc_dropout']
BayesianMode = Literal['laplace', 'mc_dropout']

AVAILABLE_SCRIPT_PRUNING_STRATEGIES = pruning_wrapper.AVAILABLE_PRUNING_STRATEGIES + ['all']