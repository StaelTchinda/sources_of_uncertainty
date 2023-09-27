from typing import Any, Callable, Dict, Literal, Optional, Text, Tuple, Type, TypeVar, Union, Collection
import decimal


import torch
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector

T = TypeVar('T')  # assume that T can only be GenericA or GenericB


__DEBUG_MODE__ = __debug__ # False

def check_is(expected: T, actual: T,
                  message: Optional[Union[Text, Callable[[], Text]]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return
    def default_message() -> Text:
        return f"Expected {expected} at id {id(expected)}, but got {actual} at id {id(expected)}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if expected is not actual:
        raise error_type(get_message())
    

def check_equals(expected: T, actual: T,
                  message: Optional[Union[Text, Callable[[], Text]]] = None,
                  are_equals: Optional[Callable[[T, T], bool]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if are_equals is None:
        are_equals = lambda x, y: x == y

    def default_message() -> Text:
        return f"Expected {expected}, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not are_equals(expected, actual):
        raise error_type(get_message())


def check_not_equals(not_expected: T, actual: T,
                      message: Optional[Union[Text, Callable[[], Text]]] = None,
                      are_equals: Optional[Callable[[T, T], bool]] = None,
                      error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if are_equals is None:
        are_equals = lambda x, y: x == y

    def default_message() -> Text:
        return f"Did not expected {not_expected}, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if are_equals(not_expected, actual):
        raise error_type(get_message())


def check_none(actual,
                message: Optional[Union[Text, Callable[[], Text]]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    def default_message() -> Text:
        return f"Expected None, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if actual is not None:
        raise error_type(get_message())


def check_not_none(actual,
                    message: Optional[Union[Text, Callable[[], Text]]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    def default_message() -> Text:
        return f"Did not expected None, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if actual is None:
        raise error_type(get_message())


def check_not_nan(tensor: torch.Tensor,
                   max_diff: int = 100,
                   message: Optional[Union[Text, Callable[[], Text]]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    is_nan = torch.isnan(tensor)

    def default_message() -> Text:
        diff_indices = torch.nonzero(is_nan, as_tuple=False)
        return f"Expected tensor of shape {tensor.shape} to not contain any nan but found {diff_indices.size(0)} nan values at indices " + ", ".join([str(i) for i in diff_indices[:max_diff].flatten().tolist()]) + ("..." if diff_indices.size(0) > max_diff else "")

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not (~is_nan).all():
        raise error_type(get_message())

def check_contains(expected: Collection[T], actual: T,
                    message: Optional[Union[Text, Callable[[], Text]]] = None,
                    contains: Optional[Callable[[Collection[T], T], bool]] = None,
                    error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if contains is None:
        contains = lambda collection, element: element in collection

    def default_message() -> Text:
        return f"Expected one element of {expected}, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not contains(expected, actual):
        raise error_type(get_message())


def check_not_contains(expected: Collection[T], actual: T,
                    message: Optional[Union[Text, Callable[[], Text]]] = None,
                    contains: Optional[Callable[[Collection[T], T], bool]] = None,
                    error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if contains is None:
        contains = lambda collection, element: element in collection

    def default_message() -> Text:
        return f"Expected one element of {expected}, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if contains(expected, actual):
        raise error_type(get_message())

def check_contains_any(expected: Collection[T], actual: Collection[T],
                message: Optional[Union[Text, Callable[[], Text]]] = None,
                contains: Optional[Callable[[Collection[T], Collection[T]], bool]] = None,
                error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if contains is None:
        contains = lambda collection, subcollection: any([element in collection for element in subcollection])

    def default_message() -> Text:
        return f"Expected one element of {expected}, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not contains(expected, actual):
        raise error_type(get_message())


def check_contains_all(expected: Collection[T], actual: Collection[T],
                message: Optional[Union[Text, Callable[[], Text]]] = None,
                contains: Optional[Callable[[Collection[T], Collection[T]], bool]] = None,
                error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if contains is None:
        contains = lambda collection, subcollection: all([element in collection for element in subcollection])

    def default_message() -> Text:
        return f"Expected one element of {expected}, but got {actual}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not contains(expected, actual):
        raise error_type(get_message())


def check_state_dict_equals(expected: Dict[Text, torch.Tensor], actual: Dict[Text, torch.Tensor]):
    if not __debug__ or not __DEBUG_MODE__:
        return

    check_equals(expected.keys(), actual.keys(), f"Expected keys: {expected.keys()}, but got keys: {actual.keys()}")
    for key in expected.keys():
        try:
            check_tensor_close(
                expected[key],
                actual[key],
                rtol=0,
                atol=0
            )
        except AssertionError as e:
            raise AssertionError(f"Inconsistency at key {key}: {e}")

def check_model_vector_equals(expected: nn.Module, actual: nn.Module, error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    check_tensor_close(
        parameters_to_vector(expected.parameters()),
        parameters_to_vector(actual.parameters()),
        rtol=0,
        atol=0,
        error_type=error_type
    )

def check_model_parameters_equals(expected: nn.Module, actual: nn.Module, error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    check_equals(
        expected.state_dict().keys(),
        actual.state_dict().keys(),
        error_type=error_type
    )
    for key in expected.state_dict().keys():
        try:
            check_tensor_close(
                expected.state_dict()[key],
                actual.state_dict()[key],
                rtol=0,
                atol=0,
                error_type=error_type
            )
        except error_type as e:
            raise error_type(f"Inconsistency at key {key}: {e}")


def check_le(actual: T, expected_max: T,
              message: Optional[Union[Text, Callable[[], Text]]] = None,
              compare: Optional[Callable[[T, T], float]] = None,
              atol: float = 0,
              error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if compare is None:
        compare = lambda a, b: (a - b)

    error_detail: Text = ''
    if atol!=0:
        error_detail = f" + {atol}, but got an error of {compare(actual, expected_max)}"

    def default_message() -> Text:
        return f"Expected {actual} < {expected_max}{error_detail}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not compare(actual, expected_max) <= atol:
        raise error_type(get_message())


def check_lt(actual: T, expected_max: T,
              message: Optional[Union[Text, Callable[[], Text]]] = None,
              compare: Optional[Callable[[T, T], float]] = None,
              error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    if compare is None:
        compare = lambda a, b: (a > b) - (a < b)

    def default_message() -> Text:
        return f"Expected {actual} < {expected_max}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not compare(actual, expected_max) < 0:
        raise error_type(get_message())


def check_is_instance(object: Any, expected_type: type, message: Optional[Union[Text, Callable[[], Text]]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    def default_message() -> Text:
        return f"Expected type {expected_type}, but got {type(object)}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    if not isinstance(object, expected_type):
        raise error_type(get_message())


def check_check_message(default_message: Callable[[], Text], message: Optional[Union[Text, Callable[[], Text]]]) -> \
        Callable[[], Text]:
    if message is None:
        get_message = default_message
    elif isinstance(message, Text):
        get_message = lambda: message
    elif isinstance(message, Callable):
        get_message = message
    else:
        raise ValueError(f"Invalid type for assertion message: {type(message)}")
    return get_message


def check_tensor_all_fulfill(x: torch.Tensor, condition: Callable[[float], bool], max_diff: int = 100,
                             error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    y = torch.empty_like(x)
    updated_condition = lambda x, y: condition(x)

    fulfill = torch.Tensor.map_(x, y, updated_condition)

    def _print_index_failure(diff_index: torch.Tensor) -> Text:
        tuple_idx: Tuple = tuple(diff_index.tolist())
        return f"{tuple_idx}: {x[tuple_idx]}"

    def _get_message() -> Text:
        failure_indices = torch.nonzero(~fulfill, as_tuple=False)
        return f"Expected tensor to fulfill condition but it does not at {failure_indices.size(0)} indices " + " \n\t ".join(
            [f"{_print_index_failure(diff_index)}, " for diff_index in failure_indices[:max_diff]]) + (
            "..." if failure_indices.size(0) > max_diff else "")

    if not fulfill.all():
        raise error_type(_get_message())


def check_tensors_all_fulfill(x: torch.Tensor, y: torch.Tensor, condition: Callable[[float, float], bool], max_diff: int = 100,
                              error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    fulfill = torch.Tensor.map_(x, y, condition)

    def _print_index_failure(diff_index: torch.Tensor) -> Text:
        tuple_idx: Tuple = tuple(diff_index.tolist())
        return f"{tuple_idx}: x = {x[tuple_idx]} and y = {y[tuple_idx]}"

    def _get_message() -> Text:
        failure_indices = torch.nonzero(~fulfill, as_tuple=False)
        return f"Expected tensors to fulfill condition but they do not at {failure_indices.size(0)} indices " + " \n\t ".join(
            [f"{_print_index_failure(diff_index)}, " for diff_index in failure_indices[:max_diff]]) + (
            "..." if failure_indices.size(0) > max_diff else "")

    if not fulfill.all():
        raise error_type(_get_message())


def check_tensor_close(x: torch.Tensor, y: torch.Tensor, max_diff: int = 100, rtol: float = 1e-05,
                        atol: float = 1e-08,
                        error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    check_tensor_same_shape(x, y)

    are_close: torch.Tensor = torch.isclose(x, y, atol=atol, rtol=rtol)

    def _print_index_diff(diff_index: torch.Tensor) -> Text:
        tuple_idx: Tuple = tuple(diff_index.tolist())
        return f"{tuple_idx}: {x[tuple_idx]} vs {y[tuple_idx]}"

    def _get_message() -> Text:
        diff_indices = torch.nonzero(~are_close, as_tuple=False)
        return f"Expected both tensors to be equals but they are different at {diff_indices.size(0)} indices " + " \n\t ".join(
            [f"{_print_index_diff(diff_index)}, " for diff_index in diff_indices[:max_diff]]) + (
            "..." if diff_indices.size(0) > max_diff else "")

    if not are_close.all():
        raise error_type(_get_message())


def check_float_close(x: float, y: float, rtol: float = 1e-05, atol: float = 1e-08, to_round: Union[bool, int] = False):
    if not __debug__ or not __DEBUG_MODE__:
        return

    final_y: float = y
    final_x: float = x
    if isinstance(to_round, bool) and to_round:
        final_y = round(y, -decimal.Decimal(str(x)).as_tuple().exponent)
    elif isinstance(to_round, int) and to_round:
        final_x = round(x, to_round)
        final_y = round(y, to_round)
    check_tensor_close(torch.tensor([final_x], dtype=torch.float64), torch.tensor([final_y], dtype=torch.float64), rtol=rtol, atol=atol)

def check_tensor_same_shape(x: torch.Tensor,
                             y: torch.Tensor,
                             message: Optional[Union[Text, Callable[[], Text]]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    def default_message() -> Text:
        return f"Expected both tensors to have same shape, but found {x.shape} vs {y.shape}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    check_equals(x.shape, y.shape, message=get_message)

def check_throws(f: Callable[[], Any], expected_exception: Type[Exception] = Exception, message: Optional[Union[Text, Callable[[], Text]]] = None,
                  error_type: Type[Exception] = ValueError):
    if not __debug__ or not __DEBUG_MODE__:
        return

    def default_message() -> Text:
        return f"Expected function {f} to throw {expected_exception}"

    def get_message() -> Text:
        return check_check_message(default_message, message)()

    try:
        f()
    except expected_exception:
        return
    except Exception as e:
        raise AssertionError(f"Expected function {f} to throw {expected_exception}, but got {e}")
    raise AssertionError(get_message())
