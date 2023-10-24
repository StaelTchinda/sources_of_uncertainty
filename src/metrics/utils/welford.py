from typing import List, Iterable, Optional, Union, Tuple, Text

import torch

from util.assertion import assert_tensor_close

# Inspired from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm


_WelfordAggregate = Tuple[int, torch.Tensor, torch.Tensor]

def welford_algorithm(elements: Iterable[torch.Tensor], expected_shape: Optional[torch.Size] = None) -> Union[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    existing_aggregate: _WelfordAggregate
    if expected_shape is not None:
        existing_aggregate = (0, torch.zeros(expected_shape), torch.zeros(expected_shape))
    else:
        existing_aggregate = (0, torch.tensor([]), torch.tensor([]))
    for element in elements:
        if expected_shape is None and existing_aggregate[0] == 0:
            existing_aggregate = initialize(element)
        else:
            existing_aggregate = update(existing_aggregate, element)

    print(f"existing_aggregate: {existing_aggregate}")

    return finalize(existing_aggregate)

def initialize(new_value: torch.Tensor) -> _WelfordAggregate:
    existing_aggregate:_WelfordAggregate = (0, torch.zeros_like(new_value), torch.zeros_like(new_value))
    existing_aggregate = update(existing_aggregate, new_value)
    return existing_aggregate

# For a new value new_value, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existing_aggregate: _WelfordAggregate, new_value: torch.Tensor) -> _WelfordAggregate:
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, biased variance and unbiased variance from an aggregate
def finalize(existing_aggregate: _WelfordAggregate):
    (count, mean, M2) = existing_aggregate
    if count < 1:
        raise ValueError(f"count must be greater than or equal to 1, but got {count}")
    elif count == 1:
        return (mean, M2 / count, M2 / count) # M2 should be 0, but we return it anyways
    else:
        (mean, biased_variance, unbiased_variance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, biased_variance, unbiased_variance)



def test():
    x: torch.Tensor = torch.rand(size=(100000, 20))
    torch_var: torch.Tensor = x.var(dim=0)
    torch_mean: torch.Tensor = x.mean(dim=0)

    values: List[torch.Tensor] = [x_ for x_ in x]

    result = welford_algorithm(values)

    welford_mean, welford_biased_var, welford_unbiased_var = result

    assert_tensor_close(torch_mean, welford_mean)
    assert_tensor_close(torch_var, welford_unbiased_var)


if __name__ == "__main__":
    test()