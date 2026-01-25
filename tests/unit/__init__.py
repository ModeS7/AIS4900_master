"""Unit tests - fast, isolated, heavily mocked.

These tests verify individual components in isolation:
- No real I/O or network calls
- All external dependencies mocked
- Each test runs in <1 second
- No GPU required

Run with: pytest tests/unit -v
"""
