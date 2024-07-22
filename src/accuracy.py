import numpy as np
import argparse


def accuracy(groundtruth, predicted):

    mse = np.mean((groundtruth - predicted) ** 2)
    rmse = np.sqrt(mse)

    # Calculate the root mean squared value of the ground truth grid
    gt_rms = np.sqrt(np.mean(groundtruth**2))
    print(gt_rms)
    relative_rmse = rmse / gt_rms
    print(f"relativae rmse:{relative_rmse}, MSE:{mse}")
    return


parser = argparse.ArgumentParser()
parser.add_argument("--gt", type=str, default=None, help="ground truth data")
parser.add_argument(
    "--pred",
    type=str,
    default=None,
    help="predicted data",
)
args = parser.parse_args()
# Example usage
groundtruth = np.load(args.gt)["data"]
predicted = np.load(args.pred)["data"]
print(groundtruth.shape, predicted.shape)

accuracy(groundtruth, predicted)
