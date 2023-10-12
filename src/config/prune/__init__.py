from typing import List, Union
from config import mode as mode_config

def get_default_sparsities(data_mode: mode_config.DataMode, model_mode: mode_config.ModelMode) -> Union[List[float], List[int]]:
    return [0.0 , 0.1, 0.3, 0.5, 0.7, 0.9]