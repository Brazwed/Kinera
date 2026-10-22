"""Compatibility bootstrap for test environments where pytest is not pre-installed."""

from __future__ import annotations

import sys
import types

try:
    import pytest
except ModuleNotFoundError:
    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pytest>=8.0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import pytest
    except Exception:
        class _Approx:
            def __init__(self, expected: float, abs: float = 1e-5):
                self.expected = expected
                self.abs = abs

            def __eq__(self, actual: object) -> bool:
                if not isinstance(actual, (int, float)):
                    return False
                return abs(actual - self.expected) <= self.abs

            def __repr__(self) -> str:
                return f"approx({self.expected} ± {self.abs})"

        class _Mark:
            def parametrize(self, *args: object, **kwargs: object) -> object:
                def decorator(fn: object) -> object:
                    return fn

                return decorator

        pytest = types.ModuleType("pytest")  # type: ignore[assignment]
        pytest.approx = _Approx  # type: ignore[attr-defined]
        pytest.mark = _Mark()  # type: ignore[attr-defined]

sys.modules["pytest"] = pytest
