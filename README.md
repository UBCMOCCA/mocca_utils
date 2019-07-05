# mocca_utils

`mocca_utils` is a Python module intended for improving the quality of life of MOCCA members.  `plots.visplot` contains commonly used plots such as scatter and time-series (rolling window) plots.

The library is built on top of [vispy](http://vispy.org/), a more performant but lesser known alternative to matplotlib.  The main feature is that the plots are designed to easily integrate with main loop of another animation or simulation script.

<img title="Demo" style="width: 80%; display: block; margin: 2em auto 2em auto; border-radius: 25px;" src="https://raw.githubusercontent.com/UBCMOCCA/mocca_utils/assets/assets/mocca_utils_demo.gif">

***

## [Installation](#installation)

### Pip

```shell
pip install git+https://github.com/UBCMOCCA/mocca_utils#egg=mocca_utils
```


## [Usage Overview](#usage-overview)

Some days, all you want is a blank figure on your desktop to look at.

```python
from mocca_utils.plots.visplot import Figure

fig = Figure()

# Events stacks up in a queue
# They are not processed until `redraw` is called
while True:    
    fig.redraw()
```

***

Then, draw a heart using `ArrowPlot` to fill the blank canvas.

```python
import numpy as np

from mocca_utils.plots.visplot import Figure, ArrowPlot

fig = Figure()

num_points = 1000
red = [1, 0, 0, 1]

plot = ArrowPlot(
    figure=fig,
    xlim=[-20, 20],
    ylim=[-20, 20],
    plot_options={
        "width": 3,
        "arrow_size": 15,
        "arrow_color": [red],
        "color": np.repeat([red], num_points, axis=0),
    },
)

t = np.linspace(0, 2*np.pi, num_points).reshape(num_points, 1)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

curve = np.expand_dims(np.concatenate((x, y), axis=1), axis=0)

while True:
    curve = np.roll(curve, 1, axis=1)
    arrow = np.concatenate((curve[:, -1, :], curve[:, -2, :]), axis=1)
    plot.update(curve, arrow)
    fig.redraw()
```

***

Multiple plots can be stacked on top of one another.  An easier way would be to stack the curves instead of the plots, but this shows plots can be composed.

```python
import numpy as np

from mocca_utils.plots.visplot import Figure, ArrowPlot

fig = Figure()

num_points = 1000
red = [1, 0, 0, 1]
green = [0, 1, 0, 1]

plot1 = ArrowPlot(
    figure=fig,
    xlim=[-20, 20],
    ylim=[-20, 20],
    plot_options={
        "width": 3,
        "arrow_size": 15,
        "arrow_color": [red],
        "color": np.repeat([red], num_points, axis=0),
    },
)

t = np.linspace(0, 2*np.pi, num_points).reshape(num_points, 1)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

mystery1 = np.expand_dims(np.concatenate((x, y), axis=1), axis=0)

plot2 = ArrowPlot(
    figure=None,
    plot_options={
        "parent": plot1.view.scene,
        "width": 3,
        "arrow_size": 15,
        "arrow_color": [green],
        "color": np.repeat([green], num_points, axis=0),
    },
)

t = np.linspace(0, 12*np.pi, num_points).reshape(num_points, 1)
x = 2 * np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4*t) - np.sin(t/12) ** 5)
y = 2 * np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4*t) - np.sin(t/12) ** 5)

mystery2 = np.expand_dims(np.concatenate((x, y), axis=1), axis=0)
arrow2 = np.concatenate((mystery2[:, -1, :], mystery2[:, -2, :]), axis=1)
plot2.update(mystery2, arrow2)

while True:    
    mystery1 = np.roll(mystery1, 1, axis=1)
    arrow1 = np.concatenate((mystery1[:, -1, :], mystery1[:, -2, :]), axis=1)
    plot1.update(mystery1, arrow1)
    fig.redraw()
```

***

Complete code for the demo above.  Most configuration options are directly passed to vispy, so you should have as much customizability as vispy provides.

```python
from mocca_utils.plots.visplot import *

fig = Figure(nrows=3, ncols=3)

# rows and cols can be (both) integers, slices, or None
# if None, then will occupy any available space of size 1
# x-axis range is defined by window_size and xlim
# window_size also defines the number of points we are plotting
# as long as it is constant (and small-ish), we can plot relatively fast
T = 100
lp1 = TimeSeriesPlot(
    figure=fig,
    tile_rows=slice(0, 1),
    tile_cols=slice(0, 2),
    num_lines=2,
    # x_axis_options={},
    y_axis_options={},
)
lp1_options = {"color": (1, 0, 0, 1)}

# these plots can be a bit slow if you have many series
# MultiTimeSeriesPlot provides a more convenient and faster time-series plot
mts = MultiTimeSeriesPlot(
    figure=fig,
    tile_rows=slice(1, 2),
    tile_cols=slice(0, 3),
    ts_rows=3,
    ts_cols=3,
    window_size=40,
)

# for scatter plot, we always take 3D points
# but we can choose to visualize in 3d or 2d using `projection`
sc1 = ScatterPlot(figure=fig, projection="3d")

# Turning off axis will make plotting faster
sc2 = ScatterPlot(
    figure=fig,
    tile_rows=slice(2, 3),
    tile_cols=slice(1, 3),
    x_axis_options={},
    y_axis_options={},
)

pos = np.random.normal(size=(10000, 3), scale=1.2)
pos = np.matmul(pos, [[0.7, 0, 0.3], [0, 1, 0], [0.6, 0, 0.4]]) / 20
rgba = np.random.uniform(0, 1, size=(10000, 4))
options = {"face_color": rgba}
sc1.update(pos, options=options)

# Generate line segments
points_per_path = 100
theta = np.linspace(0, 2 * np.pi, points_per_path).reshape(points_per_path, 1)
circle = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
# (N, T, 2) - N paths, T timesteps in each path, and 2D
paths = np.stack((circle + 0.5, circle - 0.5), axis=0).astype(np.float32)

# Colormap lets us convert float between 0 and 1 to a colour
summer_colours = vispy.color.get_colormaps()["summer"]
arrow_colours = summer_colours.map(np.linspace(0, 1, paths.shape[0]))

# ArrowPlot are line plots with a head marker
ar1 = ArrowPlot(
    figure=fig,
    xlim=[-2, 2],
    ylim=[-2, 2],
    plot_options={
        "width": 3,
        "arrow_size": 10,
        "arrow_color": arrow_colours,
        # colour needs to be defined for each vertex
        "color": np.repeat(arrow_colours, points_per_path, axis=0),
    },
)

arrows = np.concatenate((paths[:, -2, :], paths[:, -1, :]), axis=1)
ar1.update(paths, arrows)

num_paths = 2
path_length = 100

autumn_colours = vispy.color.get_colormaps()["autumn"]
arrow_colours = autumn_colours.map(np.linspace(0, 1, num_paths))

N = num_paths * path_length
connect = np.empty((N, 2), np.int32)
connect[:, 0] = np.arange(N)
connect[:, 1] = connect[:, 0] + 1
lasts = (np.arange(num_paths) + 1) * path_length - 1
connect[lasts, 1] = connect[lasts, 0]

# Can plot multiple plots in the same space
# need to specify figure=None and plot_options.parent
ar2 = ArrowPlot(
    figure=None,
    xlim=[-2, 2],
    ylim=[-2, 2],
    plot_options={
        "parent": ar1.view.scene,
        "width": 3,
        "arrow_size": 10,
        "connect": connect,
        "arrow_color": arrow_colours,
        # colour needs to be defined for each vertex
        "color": np.repeat(arrow_colours, path_length, axis=0),
    },
)
trajectories = np.zeros(shape=(num_paths, path_length, 2), dtype=np.float32)

x = 0
while True:
    x += 1
    y1 = np.sin(2 * np.pi * x / (T * 20)) * np.sin(2 * np.pi * x / T)
    y2 = np.sin(2 * np.pi * x / (T * 20)) * np.cos(2 * np.pi * x / T)

    lp1.add_point(y1, 0, options=lp1_options)
    lp1.add_point(y2, 1)

    # Drawing once at the end will redraw every plot
    pos = np.random.normal(size=(10000, 3), scale=1.2)
    sc2.update(pos, options=options)

    mts.add_points(np.random.randn(3, 3))

    trajectories[:, :-1] = trajectories[:, 1:]  # roll
    trajectories[0:, -1] = [y1, y2]
    trajectories[1:, -1] = [y2, y1]

    arrows = np.concatenate((trajectories[:, -2], trajectories[:, -1]), axis=1)
    ar2.update(trajectories, arrows)

    # you need to either add the points with redraw=True
    # or you can explicitly call redraw on the figure or the plots
    fig.redraw()
```

## Bugs

This module is designed to be fast.  It is a bug if you can't do things easily, or if things are running slow!  In case if you find a bug, there is no way to report it, just fix it yourself and submit a pull request / merge directly to master.  Your efforts will be appreciated by everyone!