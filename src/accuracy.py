import numpy as np


def relative_rmse(groundtruth, predicted):
    """
    Calculate the relative RMSE between groundtruth and predicted grids.

    Parameters:
    groundtruth (ndarray): Ground truth grid.
    predicted (ndarray): Predicted grid.

    Returns:
    float: Relative RMSE.
    """
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((groundtruth - predicted) ** 2)

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Calculate the root mean squared value of the ground truth grid
    groundtruth_rms = np.sqrt(np.mean(groundtruth**2))

    # Calculate the relative RMSE
    relative_rmse = rmse / groundtruth_rms

    return relative_rmse


# Example usage
groundtruth = np.load("/home/stan/data/pinn/downsampled_data.npy")
predicted = np.load("/home/stan/data/pinn/output/predicted_data.npz")

rel_rmse = relative_rmse(groundtruth, predicted["data"])
print(f"Relative RMSE: {rel_rmse}")
