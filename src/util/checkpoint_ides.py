import io
import os
import pickle
import dill
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Callable, Literal, Text, Optional, Any, Dict

import torch
from torch import nn

CHECKPOINTS_PATH: Path = (Path(__file__).parent.parent.parent.parent / "checkpoints").resolve()
CHECKPOINTS_MODEL_PATH: Path = CHECKPOINTS_PATH / "model"
CHECKPOINTS_HESSIAN_PATH: Path = CHECKPOINTS_PATH / "hessian"
CHECKPOINTS_HESSIAN_MATRIX_PATH: Path = CHECKPOINTS_HESSIAN_PATH / "matrix"
CHECKPOINTS_HESSIAN_IMPROVEMENT_PATH: Path = CHECKPOINTS_HESSIAN_PATH / "improvement"
CHECKPOINTS_TENSORBOARD_PATH = CHECKPOINTS_PATH / "tensorboard"

def _update_path_in_kwargs(new_path, kwargs: Dict[Text, Any]):
    if 'save_path' not in kwargs:
        kwargs['save_path'] = new_path

def find_pretrained(log_path: Path) -> Optional[Path]:
    if not log_path.exists():
        return None
    pretrained = list(log_path.glob("pretrained/model*.pt"))
    # print(f"Found at Log path: {log_path} {len(checkpoints)} checkpoints: {checkpoints}")
    if len(pretrained) == 0:
        return None
    return pretrained[-1]

def find_best_checkpoint(log_path: Path) -> Optional[Path]:
    if not log_path.exists():
        return None
    checkpoints = list(log_path.glob("lightning_logs/version_*/checkpoints/*-best.ckpt"))
    # print(f"Found at Log path: {log_path} {len(checkpoints)} checkpoints: {checkpoints}")
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]

def save_model(model: nn.Module,
               file_name: Text,
               **kwargs) -> Path:
    _update_path_in_kwargs(CHECKPOINTS_MODEL_PATH, kwargs)
    return save_torch_object(model.state_dict(), file_name, **kwargs)


def save_tensor(tensor: torch.Tensor,
                file_name: Text,
                **kwargs) -> Path:
    return save_torch_object(tensor, file_name, **kwargs)


def save_torch_object(torch_object: Any,
                      file_name: Text,
                      file_ext: Text = 'pt',
                      add_timestamp_to_name: bool = True,
                      save_path: Optional[Path] = CHECKPOINTS_PATH) -> Path:
    if add_timestamp_to_name:
        file_name = file_name + " " + datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    file_path: Path = save_path / f"{file_name}.{file_ext}"
    torch.save(torch_object, file_path)
    return file_path


def save_object(object: Any,
                file_name: Text,
                file_ext: Text = 'pt',
                add_timestamp_to_name: bool = True,
                save_path: Optional[Path] = CHECKPOINTS_PATH,
                library: Literal["dill", "pickle"] = 'pickle',
                dump_kwargs: Optional[Dict[Text, Any]]=None) -> Path:
    if add_timestamp_to_name:
        file_name = file_name + " " + datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    file_path: Path = save_path / f"{file_name}.{file_ext}"
    if library == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(object, f, **(dump_kwargs if dump_kwargs is not None else {}) )
    elif library == 'dill':
        with open(file_path, 'wb') as f:
            # IDEA: how to speed up the dumping and loading of the hessian?
            # use a different protocol 5, instead of 4, the default one
            # set byref to True
            import sys
            print(f"Dumping with dill with memory usage {sys.getsizeof(object)/8*(1024 ** 2)} MB for object class {object.__class__} at path {file_path}")
            import dill.detect
            with dill.detect.trace():
                dill.dump(object, f, **(dump_kwargs if dump_kwargs is not None else {}) )
    else:
        raise ValueError(f"Unknown library {library}")
    return file_path


def load_model(model: nn.Module,
               file_name: Text,
               update_state_dict: Optional[Callable[[Dict], Dict]] = None,
               path_args: Dict[Text, Any] = {}, load_args: Dict[Text, Any] = {}) -> nn.Module:
    _update_path_in_kwargs(CHECKPOINTS_MODEL_PATH, path_args)
    state_dict = load_torch_object(file_name, path_args, load_args)
    if state_dict is None:
        return None
    else:
        if update_state_dict is not None:
            state_dict = update_state_dict(state_dict)
        model.load_state_dict(state_dict)
        return model


def load_tensor(file_name: Text, 
                path_args: Dict[Text, Any] = {}, load_args: Dict[Text, Any] = {}) -> torch.Tensor:
    return load_torch_object(file_name, path_args, load_args)


def load_torch_object(file_name: Text, path_args: Dict[Text, Any] = {}, load_args: Dict[Text, Any] = {}) -> Any:
    file_path = get_file_path(file_name, **path_args)
    if file_path is not None:
        return torch.load(file_path, **load_args)
    else:
        return None


def load_object(file_name: Text, path_args: Dict[Text, Any] = {}, load_args: Dict[Text, Any] = {},
                library: Literal["dill", "pickle"] = 'pickle') -> Any:
    file_path = get_file_path(file_name, **path_args)
    if file_path is not None:
        if library == 'pickle':
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f, **load_args)
            except Exception as e:
                with open(file_path, 'rb') as f:
                    return CPU_Unpickler(f).load()
        elif library == 'dill':
            with open(file_path, 'rb') as f:
                # IDEA: how to speed up the dumping and loading of the hessian?
                # use a different protocol 5, instead of 4, the default one
                # set byref to True
                import sys
                print(f"Dumping with dill with memory usage {sys.getsizeof(object)/8*(1024 ** 2)} MB for object class {object.__class__} at path {file_path}")
                import dill.detect
                with dill.detect.trace():
                    return dill.load(f, **load_args )
        else:
            raise ValueError(f"Unknown library {library}")
    else:
        return None


def get_file_path(file_name: Text,
                  file_ext: Text = 'pt',
                  timestamp: Text = "last",
                  fail_if_not_found: bool = False,
                  save_path: Path = CHECKPOINTS_PATH) -> Optional[Path]:
    if timestamp == "last":
        search_path: Path = save_path / f"{file_name}*.{file_ext}"
        relevant_files = glob(str(search_path))
        if len(relevant_files) == 0:
            if fail_if_not_found:
                raise FileNotFoundError(
                    f"There exists no file with name starting  with '{file_name}' and extension '{file_ext}' in path '{save_path}'.")
            else:
                return None
        last_file = max(relevant_files, key=os.path.getmtime)
        return last_file
    else:
        file_path: Path = save_path / f"{file_name}{' ' + timestamp if timestamp is not None else ''}.{file_ext}"
        if os.path.exists(file_path):
            return file_path
        else:
            if fail_if_not_found:
                raise FileNotFoundError(
                    f"There exists no file at path {file_path} with name starting  with '{file_name}', timestamp '{timestamp}' and extension '{file_ext}' in path '{save_path}'.")
            else:
                return None


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)