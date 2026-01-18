"""Test VRAM usage of feature extractors with and without torch.compile."""

import gc
import torch


def get_vram_gb() -> float:
    """Get current VRAM usage in GB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9


def get_vram_reserved_gb() -> float:
    """Get reserved VRAM in GB (includes cached allocations)."""
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 1e9


def reset_cuda():
    """Reset CUDA memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_resnet50_default(batch_size: int = 16):
    """Test ResNet50 VRAM usage with default settings (no compile_model arg)."""
    from medgen.metrics.feature_extractors import ResNet50Features

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"ResNet50 (DEFAULT - no compile_model arg)")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    # Create extractor with default settings
    device = torch.device("cuda")
    extractor = ResNet50Features(device)

    # Force model load
    dummy_input = torch.randn(batch_size, 1, 256, 256, device=device)
    _ = extractor.extract_features(dummy_input)

    after_load = get_vram_gb()
    print(f"After load + first forward: {after_load:.2f} GB (+{after_load - baseline:.2f} GB)")

    # Run a few more forwards to trigger graph captures
    for i in range(3):
        _ = extractor.extract_features(dummy_input)

    after_warmup = get_vram_gb()
    reserved = get_vram_reserved_gb()
    print(f"After warmup: {after_warmup:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Total increase: +{after_warmup - baseline:.2f} GB")

    # Cleanup
    del extractor, dummy_input
    reset_cuda()

    return after_warmup - baseline


def test_resnet50(compile_model: bool, batch_size: int = 16):
    """Test ResNet50 VRAM usage."""
    from medgen.metrics.feature_extractors import ResNet50Features

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"ResNet50 (compile={compile_model})")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    # Create extractor
    device = torch.device("cuda")
    extractor = ResNet50Features(device, compile_model=compile_model)

    # Force model load
    dummy_input = torch.randn(batch_size, 1, 256, 256, device=device)
    _ = extractor.extract_features(dummy_input)

    after_load = get_vram_gb()
    print(f"After load + first forward: {after_load:.2f} GB (+{after_load - baseline:.2f} GB)")

    # Run a few more forwards to trigger graph captures
    for i in range(3):
        _ = extractor.extract_features(dummy_input)

    after_warmup = get_vram_gb()
    reserved = get_vram_reserved_gb()
    print(f"After warmup: {after_warmup:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Total increase: +{after_warmup - baseline:.2f} GB")

    # Cleanup
    del extractor, dummy_input
    reset_cuda()

    return after_warmup - baseline


def test_biomedclip_default(batch_size: int = 16):
    """Test BiomedCLIP VRAM usage with default settings (no compile_model arg)."""
    from medgen.metrics.feature_extractors import BiomedCLIPFeatures

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"BiomedCLIP (DEFAULT - no compile_model arg)")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    # Create extractor with default settings
    device = torch.device("cuda")
    extractor = BiomedCLIPFeatures(device)

    # Force model load
    dummy_input = torch.randn(batch_size, 1, 256, 256, device=device)
    _ = extractor.extract_features(dummy_input)

    after_load = get_vram_gb()
    print(f"After load + first forward: {after_load:.2f} GB (+{after_load - baseline:.2f} GB)")

    # Run a few more forwards to trigger graph captures
    for i in range(3):
        _ = extractor.extract_features(dummy_input)

    after_warmup = get_vram_gb()
    reserved = get_vram_reserved_gb()
    print(f"After warmup: {after_warmup:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Total increase: +{after_warmup - baseline:.2f} GB")

    # Cleanup
    del extractor, dummy_input
    reset_cuda()

    return after_warmup - baseline


def test_biomedclip(compile_model: bool, batch_size: int = 16):
    """Test BiomedCLIP VRAM usage."""
    from medgen.metrics.feature_extractors import BiomedCLIPFeatures

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"BiomedCLIP (compile={compile_model})")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    # Create extractor
    device = torch.device("cuda")
    extractor = BiomedCLIPFeatures(device, compile_model=compile_model)

    # Force model load
    dummy_input = torch.randn(batch_size, 1, 256, 256, device=device)
    _ = extractor.extract_features(dummy_input)

    after_load = get_vram_gb()
    print(f"After load + first forward: {after_load:.2f} GB (+{after_load - baseline:.2f} GB)")

    # Run a few more forwards to trigger graph captures
    for i in range(3):
        _ = extractor.extract_features(dummy_input)

    after_warmup = get_vram_gb()
    reserved = get_vram_reserved_gb()
    print(f"After warmup: {after_warmup:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Total increase: +{after_warmup - baseline:.2f} GB")

    # Cleanup
    del extractor, dummy_input
    reset_cuda()

    return after_warmup - baseline


def test_both(compile_model: bool, batch_size: int = 16):
    """Test both extractors loaded together."""
    from medgen.metrics.feature_extractors import ResNet50Features, BiomedCLIPFeatures

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"BOTH extractors (compile={compile_model})")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    device = torch.device("cuda")

    # Create both extractors
    resnet = ResNet50Features(device, compile_model=compile_model)
    biomed = BiomedCLIPFeatures(device, compile_model=compile_model)

    # Force model loads
    dummy_input = torch.randn(batch_size, 1, 256, 256, device=device)
    _ = resnet.extract_features(dummy_input)
    _ = biomed.extract_features(dummy_input)

    after_load = get_vram_gb()
    print(f"After load + first forward: {after_load:.2f} GB (+{after_load - baseline:.2f} GB)")

    # Warmup
    for i in range(3):
        _ = resnet.extract_features(dummy_input)
        _ = biomed.extract_features(dummy_input)

    after_warmup = get_vram_gb()
    reserved = get_vram_reserved_gb()
    print(f"After warmup: {after_warmup:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Total increase: +{after_warmup - baseline:.2f} GB")

    # Cleanup
    del resnet, biomed, dummy_input
    reset_cuda()

    return after_warmup - baseline


def main():
    print("Testing feature extractor VRAM usage")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    batch_size = 16

    # Test default mode first
    print("\n" + "="*60)
    print("DEFAULT MODE (current behavior)")
    print("="*60)

    resnet_default = test_resnet50_default(batch_size=batch_size)
    biomed_default = test_biomedclip_default(batch_size=batch_size)

    # Test without compile
    print("\n" + "="*60)
    print("WITHOUT torch.compile (eager mode)")
    print("="*60)

    resnet_eager = test_resnet50(compile_model=False, batch_size=batch_size)
    biomed_eager = test_biomedclip(compile_model=False, batch_size=batch_size)
    both_eager = test_both(compile_model=False, batch_size=batch_size)

    # Test with compile
    print("\n" + "="*60)
    print("WITH torch.compile (reduce-overhead mode)")
    print("="*60)

    resnet_compiled = test_resnet50(compile_model=True, batch_size=batch_size)
    biomed_compiled = test_biomedclip(compile_model=True, batch_size=batch_size)
    both_compiled = test_both(compile_model=True, batch_size=batch_size)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Default':<12} {'Eager':<12} {'Compiled':<12} {'Compiled-Eager':<15}")
    print("-"*70)
    print(f"{'ResNet50':<20} {resnet_default:>8.2f} GB  {resnet_eager:>8.2f} GB  {resnet_compiled:>8.2f} GB  {resnet_compiled - resnet_eager:>+10.2f} GB")
    print(f"{'BiomedCLIP':<20} {biomed_default:>8.2f} GB  {biomed_eager:>8.2f} GB  {biomed_compiled:>8.2f} GB  {biomed_compiled - biomed_eager:>+10.2f} GB")
    print(f"{'Both (eager)':<20} {'-':>8}     {both_eager:>8.2f} GB  {both_compiled:>8.2f} GB  {both_compiled - both_eager:>+10.2f} GB")


def test_3d_workload(compile_model: bool, num_volumes: int = 4, depth: int = 64):
    """Test VRAM with 3D volume feature extraction (simulates actual 3D training)."""
    from medgen.metrics.feature_extractors import ResNet50Features, BiomedCLIPFeatures
    from medgen.metrics.generation import extract_features_3d

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"3D Workload: {num_volumes} volumes x {depth} slices (compile={compile_model})")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    device = torch.device("cuda")

    # Create extractors
    resnet = ResNet50Features(device, compile_model=compile_model)
    biomed = BiomedCLIPFeatures(device, compile_model=compile_model)

    after_create = get_vram_gb()
    print(f"After creating extractors: {after_create:.2f} GB (+{after_create - baseline:.2f} GB)")

    # Create 3D volumes [B, C, D, H, W]
    volumes = torch.randn(num_volumes, 1, depth, 256, 256, device=device)
    after_volumes = get_vram_gb()
    print(f"After creating volumes: {after_volumes:.2f} GB (+{after_volumes - after_create:.2f} GB)")

    # Extract features (this is what happens during generation metrics)
    print("Extracting ResNet features from 3D volumes...")
    resnet_features = extract_features_3d(volumes, resnet, chunk_size=16)
    after_resnet = get_vram_gb()
    peak_resnet = torch.cuda.max_memory_allocated() / 1e9
    print(f"After ResNet extraction: {after_resnet:.2f} GB, peak: {peak_resnet:.2f} GB")

    torch.cuda.reset_peak_memory_stats()

    print("Extracting BiomedCLIP features from 3D volumes...")
    biomed_features = extract_features_3d(volumes, biomed, chunk_size=16)
    after_biomed = get_vram_gb()
    peak_biomed = torch.cuda.max_memory_allocated() / 1e9
    print(f"After BiomedCLIP extraction: {after_biomed:.2f} GB, peak: {peak_biomed:.2f} GB")

    final = get_vram_gb()
    reserved = get_vram_reserved_gb()
    print(f"Final: {final:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Total increase from baseline: +{final - baseline:.2f} GB")

    # Cleanup
    del resnet, biomed, volumes, resnet_features, biomed_features
    reset_cuda()

    return final - baseline


def test_lpips(use_compile: bool, batch_size: int = 16):
    """Test LPIPS VRAM usage."""
    from medgen.metrics.quality import _get_lpips_metric

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"LPIPS (compile={use_compile})")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    device = torch.device("cuda")

    # Get LPIPS metric (pass use_compile through cache_dir hack or test directly)
    # We need to bypass the lru_cache to test with different compile settings
    from monai.losses import PerceptualLoss
    metric = PerceptualLoss(
        spatial_dims=2,
        network_type="radimagenet_resnet50",
        pretrained=True,
    ).to(device)
    metric.eval()

    if use_compile:
        metric = torch.compile(metric, mode="reduce-overhead")

    after_load = get_vram_gb()
    print(f"After load: {after_load:.2f} GB (+{after_load - baseline:.2f} GB)")

    # Run forward pass
    dummy_input = torch.randn(batch_size, 3, 256, 256, device=device)
    dummy_target = torch.randn(batch_size, 3, 256, 256, device=device)

    with torch.no_grad():
        _ = metric(dummy_input, dummy_target)

    after_first = get_vram_gb()
    print(f"After first forward: {after_first:.2f} GB (+{after_first - baseline:.2f} GB)")

    # Warmup
    for i in range(3):
        with torch.no_grad():
            _ = metric(dummy_input, dummy_target)

    after_warmup = get_vram_gb()
    reserved = get_vram_reserved_gb()
    print(f"After warmup: {after_warmup:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Total increase: +{after_warmup - baseline:.2f} GB")

    # Cleanup
    del metric, dummy_input, dummy_target
    reset_cuda()

    return after_warmup - baseline


def test_all_compiled_models(batch_size: int = 16):
    """Test all compiled models together (simulates bravo mode)."""
    from medgen.metrics.feature_extractors import ResNet50Features, BiomedCLIPFeatures
    from monai.losses import PerceptualLoss

    reset_cuda()
    baseline = get_vram_gb()

    print(f"\n{'='*60}")
    print(f"ALL COMPILED MODELS (bravo mode simulation)")
    print(f"{'='*60}")
    print(f"Baseline VRAM: {baseline:.2f} GB")

    device = torch.device("cuda")

    # Create all models with compilation
    resnet = ResNet50Features(device, compile_model=True)
    biomed = BiomedCLIPFeatures(device, compile_model=True)
    lpips = PerceptualLoss(
        spatial_dims=2,
        network_type="radimagenet_resnet50",
        pretrained=True,
    ).to(device)
    lpips.eval()
    lpips = torch.compile(lpips, mode="reduce-overhead")

    after_create = get_vram_gb()
    print(f"After creating models: {after_create:.2f} GB (+{after_create - baseline:.2f} GB)")

    # Force model loads with forward passes
    dummy_2d = torch.randn(batch_size, 1, 256, 256, device=device)
    dummy_3ch = torch.randn(batch_size, 3, 256, 256, device=device)

    _ = resnet.extract_features(dummy_2d)
    _ = biomed.extract_features(dummy_2d)
    with torch.no_grad():
        _ = lpips(dummy_3ch, dummy_3ch)

    after_first = get_vram_gb()
    print(f"After first forwards: {after_first:.2f} GB (+{after_first - baseline:.2f} GB)")

    # Warmup
    for i in range(3):
        _ = resnet.extract_features(dummy_2d)
        _ = biomed.extract_features(dummy_2d)
        with torch.no_grad():
            _ = lpips(dummy_3ch, dummy_3ch)

    after_warmup = get_vram_gb()
    reserved = get_vram_reserved_gb()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"After warmup: {after_warmup:.2f} GB allocated, {reserved:.2f} GB reserved, {peak:.2f} GB peak")
    print(f"Total increase: +{after_warmup - baseline:.2f} GB")

    # Cleanup
    del resnet, biomed, lpips, dummy_2d, dummy_3ch
    reset_cuda()

    return after_warmup - baseline


def main_3d():
    """Test 3D workload specifically."""
    print("Testing 3D workload VRAM usage")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test with eager mode
    eager_usage = test_3d_workload(compile_model=False, num_volumes=2, depth=32)

    # Test with compile mode
    compiled_usage = test_3d_workload(compile_model=True, num_volumes=2, depth=32)

    print("\n" + "="*60)
    print("3D WORKLOAD SUMMARY")
    print("="*60)
    print(f"Eager mode:    {eager_usage:.2f} GB")
    print(f"Compiled mode: {compiled_usage:.2f} GB")
    print(f"Difference:    {compiled_usage - eager_usage:+.2f} GB")


def main_full():
    """Test everything including LPIPS."""
    print("Testing ALL compiled models (bravo mode simulation)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test LPIPS
    lpips_eager = test_lpips(use_compile=False)
    lpips_compiled = test_lpips(use_compile=True)

    print("\n" + "="*60)
    print("LPIPS SUMMARY")
    print("="*60)
    print(f"Eager:    {lpips_eager:.2f} GB")
    print(f"Compiled: {lpips_compiled:.2f} GB")
    print(f"Difference: {lpips_compiled - lpips_eager:+.2f} GB")

    # Test all together
    all_compiled = test_all_compiled_models()

    print("\n" + "="*60)
    print("TOTAL (all 3 models compiled)")
    print("="*60)
    print(f"ResNet50 + BiomedCLIP + LPIPS: {all_compiled:.2f} GB")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "3d":
            main_3d()
        elif sys.argv[1] == "full":
            main_full()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python test_extractor_vram.py [3d|full]")
    else:
        main()
