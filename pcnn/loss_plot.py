import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Function to parse the log file
def parse_log_file(log_file_path):
    loss_ini_list = []
    loss_pde_list = []
    lambda_list = []
    iter_list = []

    with open(log_file_path, "r") as file:
        for line in file:
            # Extract iteration and loss values using regex
            iter_match = re.search(r"Adam Iter (\d+)", line)
            if iter_match:
                iter_val = int(iter_match.group(1))
                iter_list.append(iter_val)

            # Extract loss_ini and loss_pde
            loss_match = re.search(
                r"loss_ini:\s([\d\.]+),\s+loss_pde:\s([\d\.]+)", line
            )
            if loss_match:
                loss_ini_list.append(float(loss_match.group(1)))
                loss_pde_list.append(float(loss_match.group(2)))

            # Extract lambda_ini1 and lambda_pde
            lambda_match = re.search(
                r"lambda_ini1:\s([\d\.]+),\s+lambda_pde:\s([\d\.]+)", line
            )
            if lambda_match:
                lambda_list.append(
                    float(lambda_match.group(1)) / float(lambda_match.group(2))
                )

    return iter_list, loss_ini_list, loss_pde_list, lambda_list


# Plotting function
def plot_results(iter_list, loss_ini, loss_pde, _lambda):
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # Plot loss_ini on the primary y-axis
    ax1.plot(iter_list, loss_ini, color="blue", label="loss_ini")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss_ini", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left")

    # Create a second y-axis to plot loss_pde
    ax2 = ax1.twinx()
    ax2.plot(iter_list, loss_pde, color="green", label="loss_pde")
    ax2.set_ylabel("Loss_pde", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="lower left")

    # Create an inset axis to plot lambda values on a third y-axis
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
    ax3.plot(
        iter_list, _lambda, color="red", label="lambda_ini / lambda_pde", linestyle="--"
    )
    ax3.set_ylabel("Lambda", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.legend(loc="upper right")

    plt.title("Loss and Lambda values over Iterations")
    plt.savefig("loss.png")
    # plt.show()


# Example usage
log_file_path = "test.log"  # Replace with the actual path to your log file
iter_list, loss_ini_list, loss_pde_list, lambda_list = parse_log_file(log_file_path)
plot_results(iter_list, loss_ini_list, loss_pde_list, lambda_list)
