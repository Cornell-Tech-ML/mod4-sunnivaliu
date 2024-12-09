# # type: ignore

# import pytest  # noqa: E402
# import numpy as np  # noqa: E402
# from numba import cuda  # noqa: E402
# from cuda_ops import conv1d, conv2d  # noqa: E402, F811

# def assert_close(array1, array2, atol=1e-6) -> None:  # noqa: ANN001
#     """Assert that two NumPy arrays are numerically close."""
#     assert np.allclose(array1, array2, atol=atol), "Mismatch between CPU and CUDA results"


# @pytest.mark.cudaTest
# def test_conv1d() -> None:
#     """Test CUDA implementation of conv1d against NumPy-based CPU results."""
#     # Input setup
#     batch, in_channels, width = 2, 3, 10
#     out_channels, kernel_width = 4, 3
#     stride, padding = 1, 1

#     input = np.random.randn(batch, in_channels, width).astype(np.float32)
#     weights = np.random.randn(out_channels, in_channels, kernel_width).astype(np.float32)

#     # CPU output
#     cpu_output = np.zeros(
#         (
#             batch,
#             out_channels,
#             (width + 2 * padding - kernel_width) // stride + 1,
#         ),
#         dtype=np.float32,
#     )
#     padded_input = np.pad(input, ((0, 0), (0, 0), (padding, padding)), mode="constant")
#     for b in range(batch):
#         for o in range(out_channels):
#             for w in range(cpu_output.shape[2]):
#                 cpu_output[b, o, w] = np.sum(
#                     padded_input[b, :, w * stride : w * stride + kernel_width] * weights[o, :, :]
#                 )

#     # CUDA output
#     input_cuda = cuda.to_device(input)
#     weights_cuda = cuda.to_device(weights)
#     cuda_output = conv1d(input_cuda, weights_cuda, stride=stride, padding=padding)

#     # Compare results
#     assert_close(cpu_output, cuda_output)


# @pytest.mark.cudaTest
# def test_conv2d() -> None:
#     """Test CUDA implementation of conv2d against NumPy-based CPU results."""
#     batch, in_channels, height, width = 1, 1, 4, 4
#     out_channels, kernel_height, kernel_width = 1, 3, 3
#     stride, padding = 1, 1

#     input = np.arange(batch * in_channels * height * width).reshape(
#         batch, in_channels, height, width
#     ).astype(np.float32)
#     weights = np.ones((out_channels, in_channels, kernel_height, kernel_width), dtype=np.float32)

#     cpu_output = np.zeros(
#         (
#             batch,
#             out_channels,
#             (height + 2 * padding - kernel_height) // stride + 1,
#             (width + 2 * padding - kernel_width) // stride + 1,
#         ),
#         dtype=np.float32,
#     )
#     padded_input = np.pad(
#         input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
#     )
#     for b in range(batch):
#         for o in range(out_channels):
#             for h in range(cpu_output.shape[2]):
#                 for w in range(cpu_output.shape[3]):
#                     cpu_output[b, o, h, w] = np.sum(
#                         padded_input[
#                             b, :, h * stride : h * stride + kernel_height, w * stride : w * stride + kernel_width
#                         ]
#                         * weights[o, :, :, :]
#                     )

#     # CUDA output
#     cuda_output = conv2d(input, weights, stride=stride, padding=padding)

#     # Debugging: Print intermediate results
#     print("Input:\n", input)
#     print("Weights:\n", weights)
#     print("CPU Output:\n", cpu_output)
#     print("CUDA Output:\n", cuda_output)

#     # Compare results
#     assert_close(cpu_output, cuda_output)
