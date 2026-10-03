"""Compatibility entrypoint for ``kinera.scripts.build_character_dataset``."""

from runpy import run_module


if __name__ == "__main__":
    run_module("kinera.scripts.build_character_dataset", run_name="__main__")
