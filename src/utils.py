import matplotlib.pyplot as plt
import numpy as np


def plot_slices(slices_gt, slices_pred, path):
    # if input is list -> to 3D ndarray
    # (slices_gt,) = slices_gt
    # (slices_pred,) = slices_pred
    if slices_pred is not None:
        slices = np.concatenate((slices_gt, slices_pred), axis=0)
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjust figsize if needed
        axs = axs.ravel()
    else:
        slices = slices_gt
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))  # Adjust figsize if needed

    for i in range(slices.shape[0]):
        axs[i].imshow(slices[i], cmap="seismic")
        axs[i].axis("off")  # Optionally turn off the axis

    plt.tight_layout()
    plt.savefig(path)


# gts, preds = [], []
# r1 = np.random.rand(500, 500)
# gts.append(r1)
# gts.append(r1)
# gts.append(r1)
# gts.append(r1)
# gts = np.stack(gts, axis=0)

# r2 = np.random.rand(500, 500)
# preds.append(r2)
# preds.append(r2)
# preds.append(r2)
# preds.append(r2)
# preds = np.stack(preds, axis=0)
