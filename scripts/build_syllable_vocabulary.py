"""Compatibility entrypoint for ``kinera.scripts.build_syllable_vocabulary``."""

from runpy import run_module

if __name__ == "__main__":
    run_module("kinera.scripts.build_syllable_vocabulary", run_name="__main__")
