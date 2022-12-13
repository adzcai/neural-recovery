import numpy as np
import os
import argparse
from dataclasses import asdict, dataclass


ALL_PLANTED = ["linear", "plain", "normalized"]
ALL_LEARNED = ["plain", "skip", "normalized"]
ALL_FORMS = ["gd", "exact", "approx", "relaxed"]


@dataclass
class Args:
    """
    Arguments for the experiment. See get_parser for details.
    """

    planted: str
    learned: str
    form: str

    n: int = 400
    d: int = 100
    k: int = None

    sigma: float = 0.0
    tol: float = 1e-4
    optw: int = None
    optx: int = 0
    cubic: bool = False
    whiten: bool = False

    seed: int = 42
    sample: int = 5

    quiet: bool = False
    save_details: bool = False
    save_folder: str = "./results/"
    cmap: str = "jet"

    epochs: int = 100
    lr: float = 2e-3
    beta: float = 1e-6
    activation: str = "relu"

    @staticmethod
    def parse():
        args = vars(get_parser().parse_args())
        return Args(**{k: v for k, v in args.items() if v is not None})

    def __post_init__(self):
        assert (
            self.planted in ALL_PLANTED and self.learned in ALL_LEARNED and self.form in ALL_FORMS
        )

        if self.k is None:
            if self.planted == "normalized" and self.form != "relaxed":
                self.k = 2
            else:
                self.k = 1

        if self.optw is None:
            if self.learned == "skip":
                self.optw = 1
            else:
                self.optw = 0

        self.cubic = self.optx in [1, 3]
        self.whiten = self.optx in [2, 3]

        if not self.quiet:
            # print an aligned table of the arguments and their values
            print("Arguments:")
            for arg in asdict(self):
                print(f"{arg:15} {getattr(self, arg)}")
            print()

    def get_save_folder(self):
        folder_path = f"{self.save_folder}learned_{self.learned}/planted_{self.planted}/form_{self.form}/trial"
        for short, arg in (
            ("n", "n"),
            ("d", "d"),
            ("w", "optw"),
            ("X", "optx"),
            ("stdev", "sigma"),
            ("sample", "sample"),
        ):
            folder_path += f"__{short}{getattr(self, arg)}"

        return folder_path + "/"


def get_parser():
    parser = argparse.ArgumentParser(description="phase transition")

    # the involved models and optimization form
    parser.add_argument(
        "planted",
        type=str,
        choices=ALL_PLANTED,
        help="planted model",
    )
    parser.add_argument(
        "learned",
        type=str,
        choices=ALL_LEARNED,
        help="learned model for recovery",
    )
    parser.add_argument(
        "form",
        type=str,
        choices=ALL_FORMS,
        help="whether to formulate optimization as convex program, GD (neural network training), or min norm (relaxed)",
    )

    # experiment parameters
    parser.add_argument("--n", type=int, help="number of sample")
    parser.add_argument("--d", type=int, help="number of dimension")
    parser.add_argument("--k", type=int, help="number of planted neurons")
    parser.add_argument("--sigma", type=float, help="noise")
    parser.add_argument("--tol", type=float, help="call it perfect recovery below this threshold")
    parser.add_argument(
        "--optw",
        type=int,
        choices=[0, 1, 2, 3],
        help="choice of W distribution. (0) Gaussian (d, k) | (1) smallest PCs of X | (2) select first k features | (3) neuron 1 is Gaussian, neuron 2 = - neuron 1",
    )
    parser.add_argument(
        "--optx",
        type=int,
        choices=[0, 1, 2, 3],
        help="choice of X distribution. 0=Gaussian, 1=Gaussian cubed, 2=whitened Gaussian, 3=whitened Gaussian cubed",
    )

    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--sample", type=int, help="number of trials per (n, d) pair")

    # plotting and meta level
    parser.add_argument("-q", "--quiet", action="store_true", help="whether to suppress error bars")
    parser.add_argument(
        "-s",
        "--save_details",
        action="store_true",
        help="whether to save training results",
    )
    parser.add_argument("-f", "--save_folder", type=str, help="path to save results")
    parser.add_argument("--cmap", type=str, help="the matplotlib cmap to use")

    # nonconvex training
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    parser.add_argument("--beta", type=float, help="weight decay parameter")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "sigmoid", "tanh", "gelu"],
        help="activation function",
    )

    return parser


def check_folder(path):
    if not os.path.exists(path):
        print("Creating folder: {}".format(path))
        os.makedirs(path)


def save_results(records: dict, save_folder=Args.save_folder):
    check_folder(save_folder)
    for prop, value in records.items():
        save_path = save_folder + prop
        np.save(save_path, value)
        print("Saved " + prop + " to " + save_path + ".npy")
