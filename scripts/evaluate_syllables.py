"""Compatibility entrypoint for ``kinera.scripts.evaluate_syllables``."""

from runpy import run_module

if __name__ == "__main__":
    run_module("kinera.scripts.evaluate_syllables", run_name="__main__")
