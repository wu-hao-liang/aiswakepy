"""Optional CLI entry point for batch runs.

Usage:
    uv run python main.py --config config.json
    uv run python main.py --config config.json --stage filter
    uv run python main.py --config config.json --stage filter depth wave impact viz
"""

import argparse

from aiswakepy.pipeline import ALL_STAGES, run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ShipwakeAIS — ship-wake wave height calculation"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to config.json (or inline JSON string)"
    )
    parser.add_argument(
        "--stage", nargs="+", choices=ALL_STAGES, default=None,
        help="Stages to run (default: all)"
    )
    args = parser.parse_args(argv)
    run_pipeline(args.config, stages=args.stage)


if __name__ == "__main__":
    main()
