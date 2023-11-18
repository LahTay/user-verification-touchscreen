import argparse


def parse_input():
    parser = argparse.ArgumentParser(description="Console API for TUV models learning.")

    # Temporary arguments showing how parsing works

    parser.add_argument(
        "--model_path",
        dest="model_path",
        action="store",
        default="models_temp\\model.model",
        help="sum the integers (default: find the max)",
    )
    parser.add_argument(
        "--lr", type=float, dest="lr", help="Learning rate", nargs=1, required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size",
        help="Batch size",
        nargs=1,
        required=True,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        dest="epoch",
        help="Number of epochs",
        nargs=1,
        required=True,
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        help="Name of the loss function used",
        nargs=1,
        required=True,
    )

    args = parser.parse_args(
        ["--lr=0.001", "--batch_size=64", "--epoch=10", "--loss=0.0001"]
    )  # Write things inside the list for testing
