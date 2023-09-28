from typing import Dict, Text, Any, Union, Optional
import logging
from pathlib import Path

def get_log_basic_config(filename: Optional[Union[Path, Text]] = None) -> Dict[Text, Any]:
    basic_config_params: Dict[Text, Any] =  { 
        "format": '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d in function %(funcName)s] %(message)s',
        "datefmt": '%Y-%m-%d:%H:%M:%S',
        "level": logging.DEBUG,
        "force": True
    }

    if filename is not None:
        if isinstance(filename, Path):
            basic_config_params["filename"] = str(filename)
        else:
            basic_config_params["filename"] = filename
    
    return basic_config_params