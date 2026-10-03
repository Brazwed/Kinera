"""Compatibility entrypoint for ``kinera.scripts.benchmark_linear_step``."""

from runpy import run_module

if __name__ == "__main__":
    run_module("kinera.scripts.benchmark_linear_step", run_name="__main__")
