"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retail-basket-prediction-analyzer", description="Retail analysts use this to compare next-item prediction accuracy between an n-gram model and a small LM, both trained on simulated shopping baskets data.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Retail Basket Prediction Analyzer is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
