"""Compatibility entrypoint for ``kinera.scripts.import_portal_lingua``."""

from runpy import run_module

if __name__ == "__main__":
    run_module("kinera.scripts.import_portal_lingua", run_name="__main__")
