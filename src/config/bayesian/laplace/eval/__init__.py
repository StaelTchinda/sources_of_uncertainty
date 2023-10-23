from typing import Dict, Text
from callbacks.keep_sample import KeepImagesCallbackContainer
from config.bayesian.laplace.eval import lenet as lenet_config
from config.mode import ModelMode


def get_callback_containers(model_mode: ModelMode) -> Dict[Text, KeepImagesCallbackContainer]:
    if model_mode == "lenet5":
        return lenet_config.get_mnist_callback_containers()
    else:
        return {}