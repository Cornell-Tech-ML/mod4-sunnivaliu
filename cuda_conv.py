# type: ignore
from numba import cuda
from .tensor import Tensor
import numpy as np

@cuda.jit
def conv1d_kernel(out, input, weights, stride, padding) -> None:  # noqa: ANN001, ANN201
    """CUDA kernel for 1D convolution."""
    b, oc, ow = cuda.grid(3)  # Batch, Output Channel, Output Width
    if b < out.shape[0] and oc < out.shape[1] and ow < out.shape[2]:
        value = 0.0
        for ic in range(input.shape[1]):  # Input Channels
            for kw in range(weights.shape[2]):  # Kernel Width
                iw = ow * stride + kw - padding
                if 0 <= iw < input.shape[2]:
                    value += input[b, ic, iw] * weights[oc, ic, kw]
        out[b, oc, ow] = value

def conv1d(input: Tensor, weights: Tensor, stride:int=1, padding:int=0) -> Tensor:
    """Perform 1D convolution using CUDA."""
    batch, in_channels, width = input.shape
    out_channels, _, kernel_width = weights.shape
    output_width = (width + 2 * padding - kernel_width) // stride + 1
    out = cuda.device_array((batch, out_channels, output_width), dtype=input.dtype)

    # Launch the kernel
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (batch + threads_per_block[0] - 1) // threads_per_block[0],
        (out_channels + threads_per_block[1] - 1) // threads_per_block[1],
        (output_width + threads_per_block[2] - 1) // threads_per_block[2],
    )
    conv1d_kernel[blocks_per_grid, threads_per_block](out, input, weights, stride, padding)

    return out


def conv1d(input: Tensor, weights: Tensor, stride:int=1, padding:int=0) -> Tensor:  # noqa: F811
    """Perform 1D convolution using CUDA."""
    batch, in_channels, width = input.shape
    out_channels, _, kernel_width = weights.shape
    output_width = (width + 2 * padding - kernel_width) // stride + 1
    out = cuda.device_array((batch, out_channels, output_width), dtype=input.dtype)

    # Launch the kernel
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (batch + threads_per_block[0] - 1) // threads_per_block[0],
        (out_channels + threads_per_block[1] - 1) // threads_per_block[1],
        (output_width + threads_per_block[2] - 1) // threads_per_block[2],
    )
    conv1d_kernel[blocks_per_grid, threads_per_block](out, input, weights, stride, padding)

    return out


@cuda.jit
def conv2d_kernel(out, input, weights, stride, padding, batch, out_channels) -> None:  # noqa: ANN001
    """CUDA kernel for 2D convolution."""
    combined_dim, oh, ow = cuda.grid(3)

    # Flatten batch and out_channels into a single dimension
    if combined_dim < batch * out_channels and oh < out.shape[2] and ow < out.shape[3]:
        b = combined_dim // out_channels  # Batch index
        oc = combined_dim % out_channels  # Output channel index

        value = 0.0
        for ic in range(input.shape[1]):  # Input channels
            for kh in range(weights.shape[2]):  # Kernel height
                for kw in range(weights.shape[3]):  # Kernel width
                    ih = int(oh * stride + kh - padding)  # Input height index
                    iw = int(ow * stride + kw - padding)  # Input width index

                    if 0 <= ih < input.shape[2] and 0 <= iw < input.shape[3]:
                        value += input[b, ic, ih, iw] * weights[oc, ic, kh, kw]

        # Write result to the output array
        out[b, oc, oh, ow] = value
  
def conv2d(input: np.ndarray, weights: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Perform 2D convolution using CUDA."""
    batch, in_channels, height, width = input.shape
    out_channels, _, kernel_height, kernel_width = weights.shape
    output_height = (height + 2 * padding - kernel_height) // stride + 1
    output_width = (width + 2 * padding - kernel_width) // stride + 1

    # Create output array on the device
    out = cuda.device_array(
        (batch, out_channels, output_height, output_width), dtype=np.float32
    )

    # Flatten batch and out_channels into a single dimension
    combined_dim = batch * out_channels

    # Define threads per block and blocks per grid
    threads_per_block = (16, 16, 1)  
    blocks_per_grid = (
        (combined_dim + threads_per_block[0] - 1) // threads_per_block[0],
        (output_height + threads_per_block[1] - 1) // threads_per_block[1],
        (output_width + threads_per_block[2] - 1) // threads_per_block[2],
    )

    # Launch the kernel
    conv2d_kernel[blocks_per_grid, threads_per_block](
        out,
        cuda.to_device(input.astype(np.float32)),
        cuda.to_device(weights.astype(np.float32)),
        int(stride),
        int(padding),
        int(batch),
        int(out_channels),
    )

    # Copy the result back to the host
    return out.copy_to_host()
