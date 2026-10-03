"""Compatibility entrypoint for ``kinera.scripts.benchmark_embedding_forward``."""

from runpy import run_module

if __name__ == "__main__":
    run_module("kinera.scripts.benchmark_embedding_forward", run_name="__main__")
