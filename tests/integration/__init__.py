"""Integration tests - real components, moderate speed.

These tests verify component interactions:
- Use real PyTorch models (small configs)
- May require GPU for some tests
- Each test runs in 1-60 seconds
- Use fixtures for temporary directories

Run with: pytest tests/integration -v
"""
