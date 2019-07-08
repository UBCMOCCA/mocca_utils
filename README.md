# mocca_utils

`mocca_utils` is a Python module intended for improving the quality of life of MOCCA members.  `plots.visplot` contains commonly used plots such as scatter and time-series (rolling window) plots.

The library is built on top of [vispy](http://vispy.org/), a more performant but lesser known alternative to matplotlib.  The main feature is that the plots are designed to easily integrate with main loop of another animation or simulation script.

<p align="center">
    <img title="Demo" src="https://raw.githubusercontent.com/UBCMOCCA/mocca_utils/assets/assets/mocca_utils_demo.gif">
</p>

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
        "parent": plot1,
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

Check out [`mocca_utils/plots/visplot.py`](https://github.com/UBCMOCCA/mocca_utils/blob/master/mocca_utils/plots/visplot.py) for more sample code.  Most configuration options are directly passed to vispy, so you should have as much customizability as vispy provides.

## Bugs

This module is designed to be fast.  It is a bug if you can't do things easily, or if things are running slow!  In case if you find a bug, there is no way to report it, just fix it yourself and submit a pull request / merge directly to master.  Your efforts will be appreciated by everyone!