import argparse
import numpy as np
from utils.plotting.roc import plot_roc_new


def main(rocs, labels, output, dataset_label, pt_min, pt_max):
    roc_list = [np.load(roc) for roc in rocs]
    plot_roc_new(
        roc_list,
        labels,
        dataset_label=dataset_label,
        pt_min=pt_min,
        pt_max=pt_max,
        output_path=output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rocs", "-r", type=str, nargs="+", help="Numpy files that should be read in."
    )
    parser.add_argument(
        "--labels", "-l", type=str, nargs="+", help="Labels that should be read in."
    )
    parser.add_argument("--output", "-o", type=str, help="Output path.")
    parser.add_argument("--dataset", "-d", type=str, help="dataset to use")
    parser.add_argument("--pt-min", type=int, help="pt_min")
    parser.add_argument("--pt-max", type=int, help="pt_max")
    args = parser.parse_args()
    print(args)

    main(args.rocs, args.labels, args.output, args.dataset, args.pt_min, args.pt_max)
