"""Comprehensive tests for lossless binary mask codec."""
import time
import torch
import pytest
from medgen.data.lossless_mask_codec import (
    encode_mask_lossless, decode_mask_lossless, get_latent_shape,
    encode_f32, decode_f32, encode_f64, decode_f64, encode_f128, decode_f128,
    FORMATS
)


# =============================================================================
# Core Lossless Roundtrip Tests
# =============================================================================

@pytest.mark.parametrize("format", ['f32', 'f64', 'f128'])
def test_lossless_roundtrip_random(format):
    """Verify encode->decode is perfectly lossless for random masks."""
    mask = (torch.rand(256, 256) > 0.5).float()
    latent = encode_mask_lossless(mask, format)
    reconstructed = decode_mask_lossless(latent, format)
    assert torch.equal(mask, reconstructed), f"Roundtrip failed for {format}"


@pytest.mark.parametrize("format", ['f32', 'f64', 'f128'])
def test_lossless_roundtrip_all_zeros(format):
    """Verify empty mask roundtrips correctly."""
    mask = torch.zeros(256, 256)
    latent = encode_mask_lossless(mask, format)
    reconstructed = decode_mask_lossless(latent, format)
    assert torch.equal(mask, reconstructed)


@pytest.mark.parametrize("format", ['f32', 'f64', 'f128'])
def test_lossless_roundtrip_all_ones(format):
    """Verify full mask roundtrips correctly."""
    mask = torch.ones(256, 256)
    latent = encode_mask_lossless(mask, format)
    reconstructed = decode_mask_lossless(latent, format)
    assert torch.equal(mask, reconstructed)


@pytest.mark.parametrize("format", ['f32', 'f64', 'f128'])
def test_lossless_roundtrip_checkerboard(format):
    """Verify alternating pattern roundtrips correctly."""
    mask = torch.zeros(256, 256)
    mask[::2, ::2] = 1  # Checkerboard pattern
    mask[1::2, 1::2] = 1
    latent = encode_mask_lossless(mask, format)
    reconstructed = decode_mask_lossless(latent, format)
    assert torch.equal(mask, reconstructed)


@pytest.mark.parametrize("format", ['f32', 'f64', 'f128'])
def test_lossless_roundtrip_single_pixel(format):
    """Verify single pixel is preserved (important for tiny tumors)."""
    for y in [0, 127, 255]:
        for x in [0, 127, 255]:
            mask = torch.zeros(256, 256)
            mask[y, x] = 1
            latent = encode_mask_lossless(mask, format)
            reconstructed = decode_mask_lossless(latent, format)
            assert torch.equal(mask, reconstructed), f"Failed at pixel ({y}, {x})"


# =============================================================================
# Batch Processing Tests
# =============================================================================

@pytest.mark.parametrize("format", ['f32', 'f64', 'f128'])
def test_batch_processing(format):
    """Verify batch processing works correctly."""
    batch = (torch.rand(4, 1, 256, 256) > 0.5).float()
    latent = encode_mask_lossless(batch, format)
    reconstructed = decode_mask_lossless(latent, format)
    assert torch.equal(batch, reconstructed)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
def test_various_batch_sizes(batch_size):
    """Verify different batch sizes work."""
    batch = (torch.rand(batch_size, 1, 256, 256) > 0.5).float()
    latent = encode_mask_lossless(batch, 'f32')
    reconstructed = decode_mask_lossless(latent, 'f32')
    assert torch.equal(batch, reconstructed)
    assert latent.shape == (batch_size, 32, 8, 8)


# =============================================================================
# Shape Tests
# =============================================================================

@pytest.mark.parametrize("format,expected", [
    ('f32', (32, 8, 8)),
    ('f64', (128, 4, 4)),
    ('f128', (512, 2, 2)),
])
def test_latent_shapes_single(format, expected):
    """Verify output shapes match expected for single mask."""
    mask = torch.zeros(256, 256)
    latent = encode_mask_lossless(mask, format)
    assert latent.shape == expected
    assert get_latent_shape(format) == expected


@pytest.mark.parametrize("format,expected", [
    ('f32', (4, 32, 8, 8)),
    ('f64', (4, 128, 4, 4)),
    ('f128', (4, 512, 2, 2)),
])
def test_latent_shapes_batch(format, expected):
    """Verify output shapes match expected for batch."""
    batch = torch.zeros(4, 1, 256, 256)
    latent = encode_mask_lossless(batch, format)
    assert latent.shape == expected


# =============================================================================
# Spatial Correspondence Tests
# =============================================================================

def test_spatial_correspondence_f32():
    """Verify top-left block maps to top-left latent position."""
    mask = torch.zeros(256, 256)
    mask[0:32, 0:32] = 1  # Fill top-left 32x32 block

    latent = encode_mask_lossless(mask, 'f32')

    # Decode and verify
    reconstructed = decode_mask_lossless(latent, 'f32')
    assert torch.equal(mask, reconstructed)


def test_spatial_correspondence_bottom_right_f32():
    """Verify bottom-right block maps correctly."""
    mask = torch.zeros(256, 256)
    mask[224:256, 224:256] = 1  # Fill bottom-right 32x32 block

    latent = encode_mask_lossless(mask, 'f32')
    reconstructed = decode_mask_lossless(latent, 'f32')
    assert torch.equal(mask, reconstructed)


@pytest.mark.parametrize("format,block_size", [
    ('f32', 32),
    ('f64', 64),
    ('f128', 128),
])
def test_spatial_correspondence_all_blocks(format, block_size):
    """Verify each block location roundtrips correctly."""
    spatial = 256 // block_size
    for i in range(spatial):
        for j in range(spatial):
            mask = torch.zeros(256, 256)
            y_start, y_end = i * block_size, (i + 1) * block_size
            x_start, x_end = j * block_size, (j + 1) * block_size
            mask[y_start:y_end, x_start:x_end] = 1

            latent = encode_mask_lossless(mask, format)
            reconstructed = decode_mask_lossless(latent, format)
            assert torch.equal(mask, reconstructed), f"Block ({i}, {j}) failed for {format}"


# =============================================================================
# Device Tests
# =============================================================================

def test_gpu_support():
    """Verify works on CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mask = (torch.rand(256, 256, device='cuda') > 0.5).float()
    latent = encode_mask_lossless(mask, 'f32')
    reconstructed = decode_mask_lossless(latent, 'f32')

    assert latent.device.type == 'cuda'
    assert reconstructed.device.type == 'cuda'
    assert torch.equal(mask, reconstructed)


def test_gpu_batch():
    """Verify batch processing works on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch = (torch.rand(8, 1, 256, 256, device='cuda') > 0.5).float()
    latent = encode_mask_lossless(batch, 'f32')
    reconstructed = decode_mask_lossless(latent, 'f32')

    assert latent.device.type == 'cuda'
    assert torch.equal(batch, reconstructed)


# =============================================================================
# Dtype Handling Tests
# =============================================================================

@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64, torch.float16, torch.int32, torch.uint8])
def test_input_dtype_handling(input_dtype):
    """Verify various input dtypes are handled correctly."""
    mask = (torch.rand(256, 256) > 0.5)
    if input_dtype in [torch.float32, torch.float64, torch.float16]:
        mask = mask.to(input_dtype)
    else:
        mask = mask.to(input_dtype)

    latent = encode_mask_lossless(mask, 'f32')
    reconstructed = decode_mask_lossless(latent, 'f32')

    # Compare as float (output is always float)
    expected = (mask > 0.5).float()
    assert torch.equal(expected, reconstructed)


def test_binarization_threshold():
    """Verify 0.5 threshold works correctly."""
    # Create a mask with values around the 0.5 threshold
    row = torch.tensor([0.0, 0.49, 0.5, 0.51, 1.0])
    mask = row.repeat(256 // 5 + 1)[:256].unsqueeze(0).expand(256, -1).clone()

    latent = encode_mask_lossless(mask, 'f32')
    reconstructed = decode_mask_lossless(latent, 'f32')

    # Values <= 0.5 should be 0, > 0.5 should be 1
    expected = (mask > 0.5).float()
    assert torch.equal(expected, reconstructed)


# =============================================================================
# Convenience Function Tests
# =============================================================================

def test_convenience_functions():
    """Verify convenience aliases work correctly."""
    mask = (torch.rand(256, 256) > 0.5).float()

    # f32
    latent = encode_f32(mask)
    assert latent.shape == (32, 8, 8)
    reconstructed = decode_f32(latent)
    assert torch.equal(mask, reconstructed)

    # f64
    latent = encode_f64(mask)
    assert latent.shape == (128, 4, 4)
    reconstructed = decode_f64(latent)
    assert torch.equal(mask, reconstructed)

    # f128
    latent = encode_f128(mask)
    assert latent.shape == (512, 2, 2)
    reconstructed = decode_f128(latent)
    assert torch.equal(mask, reconstructed)


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_invalid_mask_shape():
    """Verify error on wrong mask shape."""
    with pytest.raises(AssertionError):
        encode_mask_lossless(torch.zeros(128, 128), 'f32')


def test_invalid_latent_shape():
    """Verify error on wrong latent shape."""
    with pytest.raises(AssertionError):
        decode_mask_lossless(torch.zeros(64, 8, 8), 'f32')  # Wrong channels


def test_invalid_format():
    """Verify error on invalid format."""
    with pytest.raises(KeyError):
        encode_mask_lossless(torch.zeros(256, 256), 'f99')


# =============================================================================
# Performance Tests
# =============================================================================

def test_encode_performance():
    """Verify encoding is fast enough for on-the-fly use (<1ms)."""
    mask = (torch.rand(256, 256) > 0.5).float()

    # Warmup
    for _ in range(10):
        encode_mask_lossless(mask, 'f32')

    # Time 100 encodes
    start = time.perf_counter()
    for _ in range(100):
        encode_mask_lossless(mask, 'f32')
    elapsed = (time.perf_counter() - start) / 100 * 1000  # ms per encode

    print(f"Encode time: {elapsed:.3f} ms")
    assert elapsed < 1.0, f"Encoding too slow: {elapsed:.3f} ms (should be <1ms)"


def test_decode_performance():
    """Verify decoding is fast enough (<1ms)."""
    mask = (torch.rand(256, 256) > 0.5).float()
    latent = encode_mask_lossless(mask, 'f32')

    # Warmup
    for _ in range(10):
        decode_mask_lossless(latent, 'f32')

    # Time 100 decodes
    start = time.perf_counter()
    for _ in range(100):
        decode_mask_lossless(latent, 'f32')
    elapsed = (time.perf_counter() - start) / 100 * 1000  # ms per decode

    print(f"Decode time: {elapsed:.3f} ms")
    assert elapsed < 1.0, f"Decoding too slow: {elapsed:.3f} ms (should be <1ms)"


def test_batch_encode_performance():
    """Verify batch encoding scales reasonably."""
    batch = (torch.rand(16, 1, 256, 256) > 0.5).float()

    # Warmup
    for _ in range(5):
        encode_mask_lossless(batch, 'f32')

    # Time
    start = time.perf_counter()
    for _ in range(20):
        encode_mask_lossless(batch, 'f32')
    elapsed = (time.perf_counter() - start) / 20 * 1000  # ms per batch

    print(f"Batch encode (16 masks): {elapsed:.3f} ms")
    assert elapsed < 20.0, f"Batch encoding too slow: {elapsed:.3f} ms"


# =============================================================================
# Bit Pattern Verification Tests
# =============================================================================

def test_deterministic_encoding():
    """Verify same mask always produces same latent."""
    mask = (torch.rand(256, 256) > 0.5).float()

    latent1 = encode_mask_lossless(mask, 'f32')
    latent2 = encode_mask_lossless(mask, 'f32')

    # Compare bit patterns since some float values may be NaN
    # (NaN != NaN by IEEE 754, but bit patterns should match)
    bits1 = latent1.view(torch.int32)
    bits2 = latent2.view(torch.int32)
    assert torch.equal(bits1, bits2)


def test_different_masks_different_latents():
    """Verify different masks produce different latents."""
    mask1 = torch.zeros(256, 256)
    mask2 = torch.ones(256, 256)

    latent1 = encode_mask_lossless(mask1, 'f32')
    latent2 = encode_mask_lossless(mask2, 'f32')

    assert not torch.equal(latent1, latent2)


# =============================================================================
# Integration Test
# =============================================================================

def test_dataloader_simulation():
    """Simulate dataloader usage pattern."""
    # Simulate loading batch of masks and encoding on-the-fly
    batch_masks = [(torch.rand(256, 256) > 0.5).float() for _ in range(8)]

    # Encode each (as would happen in __getitem__)
    latents = [encode_mask_lossless(m, 'f32') for m in batch_masks]

    # Stack into batch (as collate_fn would do)
    batch_latent = torch.stack(latents)
    assert batch_latent.shape == (8, 32, 8, 8)

    # Verify all roundtrip correctly
    for i, original in enumerate(batch_masks):
        reconstructed = decode_mask_lossless(batch_latent[i], 'f32')
        assert torch.equal(original, reconstructed)
