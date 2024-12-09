# type: ignore

from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Ensure that the height and width are divisible by the kernel size
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.

    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to include kernel dimensions
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute the dimensions to bring kernel dimensions together
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5)

    # Flatten the kernel dimensions
    tiled = tiled.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Avgpool2d"""
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


# Task 4.4
reduce_max = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Argmax as tensor"""
    return input == reduce_max(input, dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        ctx.save_for_backward(input, dim)
        return reduce_max(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        input, dim = ctx.saved_values
        max_indices = argmax(input, int(dim.item()))
        return grad_output * max_indices, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Max on tensor"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    return input.exp() / input.exp().sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    max_val = max(input, dim)
    logsumexp = (input - max_val).exp().sum(dim).log()
    return input - logsumexp - max_val


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Max-pooling operation over 2D inputs."""
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=4).view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore=False) -> Tensor:  # noqa: ANN001
    """Dropput"""
    if not ignore:
        bit_tensor = rand(input.shape, input.backend) > rate
        input = bit_tensor * input
    return input
