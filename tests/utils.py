"""Shared test utilities and assertions."""
import math
import torch
from typing import Literal, Tuple


def make_mock_dataset(
    spatial_dims: int = 2,
    fmt: Literal["tuple", "dict", "seg_tuple", "seg_dict"] = "dict",
    length: int = 10,
    image_size: int = 64,
    depth: int = 16,
    num_bins: int = 7,
):
    """Build a small Dataset-like object for tests without fixture boilerplate.

    Consolidates the MockDataset pattern that was duplicated ~10+ times across
    tests/integration (finding M-15). Returns an object with `__len__` and
    `__getitem__`; NOT a torch.utils.data.Dataset subclass so it's cheap.

    Args:
        spatial_dims: 2 or 3.
        fmt: Output format per item:
            - "tuple":    `(image, seg)`
            - "dict":     `{'image': ..., 'seg': ..., 'patient': ...}`
            - "seg_tuple": `(seg, size_bins)` for seg_conditioned mode
            - "seg_dict":  `{'image': seg, 'size_bins': ...}`
        length: Number of items the dataset exposes.
        image_size: H == W (square).
        depth: D for 3D only.
        num_bins: Size-bin vector length.

    Returns:
        A dataset-like object. Items are reproducible per-index via seeded RNG.
    """
    def _shape(c: int = 1) -> tuple[int, ...]:
        if spatial_dims == 2:
            return (c, image_size, image_size)
        return (c, depth, image_size, image_size)

    class _MockDataset:
        def __len__(self) -> int:
            return length

        def __getitem__(self, idx: int):
            torch.manual_seed(idx)  # reproducible per item
            if fmt == "tuple":
                image = torch.rand(*_shape(1))
                seg = (torch.rand(*_shape(1)) > 0.5).float()
                return (image, seg)
            if fmt == "dict":
                return {
                    "image": torch.rand(*_shape(1)),
                    "seg": (torch.rand(*_shape(1)) > 0.5).float(),
                    "patient": f"patient_{idx}",
                }
            if fmt == "seg_tuple":
                seg = (torch.rand(*_shape(1)) > 0.5).float()
                size_bins = torch.randint(0, 5, (num_bins,))
                return (seg, size_bins)
            if fmt == "seg_dict":
                seg = (torch.rand(*_shape(1)) > 0.5).float()
                size_bins = torch.randint(0, 5, (num_bins,))
                return {"image": seg, "size_bins": size_bins}
            raise ValueError(f"Unknown fmt: {fmt}")

    return _MockDataset()


def assert_valid_probability(value: float, name: str = "value") -> None:
    """Assert value is a float in [0, 1]."""
    assert isinstance(value, float), f"{name} should be float, got {type(value)}"
    assert 0.0 <= value <= 1.0, f"{name} should be in [0, 1], got {value}"


def assert_valid_metric(value: float, name: str = "value", max_val: float = 100.0) -> None:
    """Assert value is a valid metric (float, finite, in range)."""
    assert isinstance(value, float), f"{name} should be float"
    assert not math.isnan(value), f"{name} is NaN"
    assert not math.isinf(value), f"{name} is infinite"
    assert 0.0 <= value <= max_val, f"{name} out of range: {value}"


def assert_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> None:
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, \
        f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_tensor_dtype(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype,
    name: str = "tensor"
) -> None:
    """Assert tensor has expected dtype."""
    assert tensor.dtype == expected_dtype, \
        f"{name} dtype {tensor.dtype} != expected {expected_dtype}"


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    name: str = "tensor"
) -> None:
    """Assert tensors are close within tolerance."""
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), \
        f"{name} differs: max diff = {(actual - expected).abs().max().item()}"


def assert_tensor_device(
    tensor: torch.Tensor,
    expected_device: torch.device,
    name: str = "tensor"
) -> None:
    """Assert tensor is on expected device."""
    # Handle device comparison (cuda:0 vs cuda)
    actual = tensor.device
    if actual.type != expected_device.type:
        raise AssertionError(
            f"{name} device {actual} != expected {expected_device}"
        )


def assert_no_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no NaN values."""
    if torch.isnan(tensor).any():
        raise AssertionError(f"{name} contains NaN values")


def assert_no_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no Inf values."""
    if torch.isinf(tensor).any():
        raise AssertionError(f"{name} contains Inf values")


def assert_tensor_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains only finite values."""
    assert_no_nan(tensor, name)
    assert_no_inf(tensor, name)


def assert_loss_valid(loss: torch.Tensor, name: str = "loss") -> None:
    """Assert loss tensor is valid (scalar, finite, non-negative)."""
    if loss.ndim != 0:
        raise AssertionError(f"{name} should be scalar, got shape {loss.shape}")
    assert_tensor_finite(loss, name)
    if loss.item() < 0:
        raise AssertionError(f"{name} should be non-negative, got {loss.item()}")


def assert_batch_shape(
    tensor: torch.Tensor,
    batch_size: int,
    channels: int,
    spatial_dims: int = 2,
    name: str = "tensor"
) -> None:
    """Assert tensor has valid batch shape [B, C, ...spatial...]."""
    expected_ndim = 2 + spatial_dims
    if tensor.ndim != expected_ndim:
        raise AssertionError(
            f"{name} should be {expected_ndim}D, got {tensor.ndim}D"
        )
    if tensor.shape[0] != batch_size:
        raise AssertionError(
            f"{name} batch size {tensor.shape[0]} != expected {batch_size}"
        )
    if tensor.shape[1] != channels:
        raise AssertionError(
            f"{name} channels {tensor.shape[1]} != expected {channels}"
        )
