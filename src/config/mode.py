

from typing import Literal
from network.pruning import pruner as pruning_wrapper

AVAILABLE_DATASETS = ['iris', 'mnist', 'cifar10', 'wildcam']
DataMode = Literal['iris', 'mnist', 'cifar10', 'wildcam']

AVAILABLE_JOINT_DATASETS = ['ambiguous_mnist', 'cifar10_c']
JointDataMode = Literal['ambiguous_mnist', 'cifar10_c']


AVAILABLE_MODELS = ['fc', 'lenet5', 'vgg11', 'vgg13', 'vgg16', 
'resnet18', 'resnet34', 'resnet50', 
'resnet20', 'resnet32', 'resnet44',
'nf_resnet18',
'wideresnet50']
ModelMode = Literal['fc', 'lenet5', 'vgg11', 'vgg13', 'vgg16', 
'resnet18', 'resnet34', 'resnet50',
'resnet20', 'resnet32', 'resnet44',
'nf_resnet18',
'wideresnet50']   

AVAILABLE_BAYESIAN_MODES = ['laplace', 'mc_dropout']
BayesianMode = Literal['laplace', 'mc_dropout']

AVAILABLE_SCRIPT_PRUNING_STRATEGIES = pruning_wrapper.AVAILABLE_PRUNING_STRATEGIES + ['all']