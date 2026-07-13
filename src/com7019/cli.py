"""Optional CLI entry point for headless notebook execution."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="COM7019 forecasting pipeline")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("_code/COM7019_25199053.ipynb"),
        help="Path to the assignment notebook",
    )
    args = parser.parse_args()
    print(f"Open and run: {args.notebook}")
    print("For full training, use Google Colab (GPU) or: docker compose run notebook")


if __name__ == "__main__":
    main()
