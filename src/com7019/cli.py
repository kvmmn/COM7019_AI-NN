"""CLI helper pointing to the assignment notebook."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="COM7019 forecasting pipeline")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebook/COM7019_25199053.ipynb"),
        help="Path to the assignment notebook",
    )
    args = parser.parse_args()
    print(f"Open and run: {args.notebook}")
    print("For full GPU training, use Google Colab or: docker compose run notebook")


if __name__ == "__main__":
    main()
