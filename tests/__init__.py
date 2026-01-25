"""MedGen Test Suite.

Organized into four categories:
- unit/: Fast, isolated tests with mocks (<1s each)
- integration/: Tests with real components (may need GPU)
- e2e/: Full pipeline tests (slow, need GPU + data)
- benchmarks/: Performance regression tests

Quick commands:
    pytest tests/unit -v                    # Unit tests only
    pytest -m "not slow and not gpu"        # Fast tests
    pytest --cov=src/medgen                 # With coverage
"""
