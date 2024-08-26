import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Plotting wave propagation")
    parser.add_argument(
        "--file", "-f", type=str, default=None, help="npy file for animation"
    )
    return parser.parse_args()


def update(figure, axis, step):
    if len(axis.images):
        axis.images[-1].colorbar.remove()

    axis.clear()

    first = None
    for wavefield, kwarg in zip(
        [data], [dict(vmin=-vmax_global, vmax=vmax_global, cmap="seismic")]
    ):
        # for wavefield, kwarg in zip([data], [dict(aspect='auto',cmap="seismic")]):
        _wavefield = wavefield[step][30:-30, 30:-30]
        im = axis.imshow(_wavefield.T, origin="lower", **kwarg)

        cb = plt.colorbar(im, ax=axis, shrink=0.75, format="%.0e")
        if first is None:
            cb.set_label("pressure (Pa)")
            first = False

    figure.canvas.draw_idle()


def update_slider(val):
    step = int(val)
    update(figure, axis, step)


def animate(step):
    update(figure, axis, step)


args = get_args()
data = np.load(args.file)["data"][300:]
print(data.shape)
vmax_global = abs(data).max()
# scl = abs(data).max()
# data = data/ scl
fname = args.file.split("/")[-1].split(".")[0]

figure, axis = plt.subplots()
plt.subplots_adjust(bottom=0.25)
axis.margins(x=0)

ax_shot = plt.axes([0.15, 0.1, 0.7, 0.03])
time_range = (0, data.shape[0] - 1)
slider = Slider(ax_shot, "time", time_range[0], time_range[1], valinit=0, valstep=1)
update_slider(0)

slider.on_changed(update_slider)
axis.slider = slider
anim = FuncAnimation(figure, animate, frames=time_range[1] + 1, interval=20)

# Set up formatting for the movie files
Writer = FFMpegWriter(fps=60, metadata=dict(artist="Me"), bitrate=1800)
savefile = os.path.join(os.getcwd(), f"{fname}.mp4")
anim.save(savefile, writer=Writer)
