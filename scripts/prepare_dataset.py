"""Build the streaming autoregressive character dataset."""

from kinera.scripts.build_character_dataset import build_character_dataset, main

__all__ = ["build_character_dataset", "main"]


if __name__ == "__main__":
    main()
