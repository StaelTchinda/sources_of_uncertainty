

from typing import Literal
from network.pruning import pruner as pruning_wrapper

AVAILABLE_DATASETS = ['iris', 'mnist', 'cifar10', 'wildcam']
DataMode = Literal['iris', 'mnist', 'cifar10', 'wildcam']


AVAILABLE_MODELS = ['fc', 'lenet5', 'vgg11', 'vgg13', 'vgg16', 'wideresnet50']
ModelMode = Literal['fc', 'lenet5', 'vgg11', 'vgg13', 'vgg16', 'wideresnet50']   


AVAILABLE_SCRIPT_PRUNING_STRATEGIES = pruning_wrapper.AVAILABLE_PRUNING_STRATEGIES + ['all']