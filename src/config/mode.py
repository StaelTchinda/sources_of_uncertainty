

from typing import Literal
from network.pruning import pruner as pruning_wrapper

AVAILABLE_DATASETS = ['iris', 'mnist', 'cifar10', 'wildcam', 'imagenet']
DataMode = Literal['iris', 'mnist', 'cifar10', 'wildcam', 'imagenet']

AVAILABLE_JOINT_DATASETS = ['ambiguous_mnist', 
                            'cifar10_c_fog'
                            ]
JointDataMode = Literal['ambiguous_mnist', 
                        'cifar10_c_fog'
                        ]


AVAILABLE_MODELS = ['fc', 'lenet5',
'resnet20', 'resnet32', 'resnet44',
'swin_t',
'wideresnet50']
ModelMode = Literal['fc', 'lenet5', 
'resnet20', 'resnet32', 'resnet44',
'swin_t',
'wideresnet50']   

AVAILABLE_BAYESIAN_MODES = ['laplace', 'mc_dropout']
BayesianMode = Literal['laplace', 'mc_dropout']

AVAILABLE_SCRIPT_PRUNING_STRATEGIES = pruning_wrapper.AVAILABLE_PRUNING_STRATEGIES + ['all']