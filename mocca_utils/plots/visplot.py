import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import time
import numpy as np
import vispy
from vispy import app, gloo, scene

from plots.shaders import VERT_SHADER, FRAG_SHADER


try:
    # On vispy < 0.6, this is faster
    # else it is capped at 60 fps
    import PyQt5
    import PyQt5.QtCore

    vispy.use("pyqt5", "gl2")
except ImportError:
    # On vispy > 0.6, this is faster
    vispy.use("pyglet", "gl2")


def event_handler(event):
    print(event, type(event))


def on_close(event):
    sys.exit(0)


class CustomPanZoomCamera(scene.PanZoomCamera):
    def __init__(self, rect, interactive):

        self.x_min = rect[0]
        self.x_max = self.x_min + rect[2]
        self.y_min = rect[1]
        self.y_max = self.y_min + rect[3]

        super().__init__(rect=rect, interactive=interactive)

    def expand_bounds(self, x=None, y=None):
        if y is not None:
            if y > self.y_max:
                delta = (y - self.y_max) * 2
                self.y_max = max(self.y_max * 2, self.y_max + delta)
            elif y < self.y_min:
                delta = (y - self.y_min) * 2
                self.y_min = min(self.y_min * 2, self.y_min + delta)

        if x is not None:
            if x > self.x_max:
                delta = (x - self.x_max) * 2
                self.x_max = max(self.x_max * 2, self.x_max + delta)
            elif x < self.x_min:
                delta = (x - self.x_min) * 2
                self.x_min = min(self.x_min * 2, self.x_min + delta)

        self.rect._pos = (self.x_min, self.y_min)
        self.rect._size = (self.x_max - self.x_min, self.y_max - self.y_min)


class Figure:
    def __init__(self, nrows=1, ncols=1, grid_options=None, **kwargs):
        self._in_use = np.zeros((nrows, ncols), dtype=np.bool)
        self.nrows = nrows
        self.ncols = ncols

        kwargs.setdefault("keys", "interactive")
        kwargs.setdefault("show", True)
        kwargs.setdefault("size", (600, 600))

        self.canvas = scene.SceneCanvas(**kwargs)
        # self.canvas.measure_fps()

        self.canvas.on_close = on_close
        # https://github.com/vispy/vispy/issues/1201
        self.canvas.native.closeEvent = on_close
        # self.canvas.events.connect(event_handler)

        grid_options = {} if grid_options is None else grid_options
        grid_options.setdefault("spacing", 0)

        self.grid = self.canvas.central_widget.add_grid(**grid_options)

    def get_viewport_position(self, row, col):
        return (
            self.canvas.physical_size[1] * (self.nrows - row) / self.nrows,
            self.canvas.physical_size[0] * col / self.ncols,
        )

    def _get_subplot(
        self, row=None, col=None, row_span=1, col_span=1, view_options=None
    ):

        if row is None and col is None:
            row, col = np.unravel_index(self._in_use.argmin(), self._in_use.shape)
            assert not self._in_use[row, col], "Oops, ran out of space to put new plot"

        view_options = {} if view_options is None else view_options
        view_options.setdefault("border_color", (0.5, 0.5, 0.5, 1))

        view = self.grid.add_view(row, col, row_span, col_span, **view_options)

        self._in_use[slice(row, row + row_span), slice(col, col + col_span)] = True

        return view, row, col

    def redraw(self):
        app.process_events()


class Plot:
    def __init__(
        self,
        figure="new",
        tile_rows=None,
        tile_cols=None,
        xlim=[-1, 1],
        ylim=[-1, 1],
        view_options=None,
        y_axis_options=None,
        x_axis_options=None,
    ):
        if figure is None:
            return

        self.figure = Figure() if figure == "new" else figure

        if isinstance(tile_rows, slice) and isinstance(tile_cols, slice):
            row, row_span = tile_rows.start, tile_rows.stop - tile_rows.start
            col, col_span = tile_cols.start, tile_cols.stop - tile_cols.start
        else:
            row = col = None
            row_span = col_span = 1

        self.view, self.row, self.col = self.figure._get_subplot(
            row, col, row_span, col_span, view_options
        )
        self.view.camera = CustomPanZoomCamera(
            rect=(xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0]),
            interactive=False,
        )
        self.row_span = row_span
        self.col_span = col_span

        if y_axis_options is not None:
            y_axis_options.setdefault("orientation", "right")
            y_axis_options.setdefault("axis_font_size", 12)
            y_axis_options.setdefault("axis_label_margin", 50)
            y_axis_options.setdefault("tick_label_margin", 5)

            self.yaxis = scene.AxisWidget(**y_axis_options)
            self.figure.grid.add_widget(
                self.yaxis, row=row, col=col, row_span=row_span, col_span=col_span
            )
            self.yaxis.link_view(self.view)

        if x_axis_options is not None:
            x_axis_options.setdefault("orientation", "top")
            x_axis_options.setdefault("axis_font_size", 12)
            x_axis_options.setdefault("axis_label_margin", 50)
            x_axis_options.setdefault("tick_label_margin", 5)

            self.xaxis = scene.AxisWidget(**x_axis_options)
            self.figure.grid.add_widget(
                self.xaxis, row=row, col=col, row_span=row_span, col_span=col_span
            )
            self.xaxis.link_view(self.view)

    def redraw(self):
        app.process_events()


class TimeSeriesPlot(Plot):
    def __init__(
        self,
        num_lines=1,
        window_size=1000,
        ylim=[-1.2, 1.2],
        plot_options=None,
        **kwargs
    ):
        super().__init__(ylim=ylim, xlim=[0, window_size], **kwargs)

        plot_options = {} if plot_options is None else plot_options

        if "parent" in plot_options:
            self.view = plot_options["parent"].view
            plot_options["parent"] = self.view.scene

        plot_options.setdefault("antialias", False)
        plot_options.setdefault("method", "gl")
        plot_options.setdefault("parent", self.view.scene)

        self.window_size = window_size
        x = np.arange(window_size)
        y = np.zeros(window_size)

        self.lines = [
            scene.visuals.Line(np.stack((x, y), axis=1), **plot_options)
            for _ in range(num_lines)
        ]
        self.steps = np.zeros(num_lines, dtype=np.int32)

    def add_point(self, y, line_num=0, options=None, redraw=False):

        self.view.camera.expand_bounds(y=y)

        line = self.lines[line_num]
        step = self.steps[line_num]

        shift = 1 if isinstance(y, (int, float)) else len(y)

        if step < self.window_size:
            self.steps[line_num] = step + shift
        else:
            # shift to left by length y
            line.pos[:-shift, 1] = line.pos[shift:, 1]
            step = self.window_size - shift

        line.pos[step : step + shift, 1] = y

        options = {} if options is None else options
        line.set_data(line.pos, **options)

        if redraw:
            self.redraw()


class ScatterPlot(Plot):
    def __init__(self, plot_options=None, projection=None, **kwargs):
        super().__init__(**kwargs)

        plot_options = {} if plot_options is None else plot_options

        if "parent" in plot_options:
            self.view = plot_options["parent"].view
            plot_options["parent"] = self.view.scene

        plot_options.setdefault("parent", self.view.scene)

        self.scatter = scene.visuals.Markers(**plot_options)

        if projection == "3d":
            self.scatter.set_data(np.zeros((1, 3)))
            self.view.camera = "turntable"
            self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        # TODO: might need to remove x/y_axis_options if 3d ...

    def update(self, points, options=None, redraw=False):

        if isinstance(self.view.camera, CustomPanZoomCamera):
            self.view.camera.expand_bounds(points[:, 0].min(), points[:, 1].min())
            self.view.camera.expand_bounds(points[:, 0].max(), points[:, 1].max())

        options = {} if options is None else options
        options.setdefault("edge_color", None)
        options.setdefault("face_color", (1, 1, 1, 0.5))
        options.setdefault("size", 5)

        self.scatter.set_data(points, **options)

        if redraw:
            self.redraw()


class MultiTimeSeriesPlot(Plot):
    def __init__(self, window_size=1000, ts_rows=1, ts_cols=1, colors=None, **kwargs):
        super().__init__(**kwargs)

        self.window_size = window_size
        self.ts_rows = ts_rows
        self.ts_cols = ts_cols

        # Number of signals.
        self.num_series = ts_rows * ts_cols

        # Various signal amplitudes.
        # amplitudes = 0.1 + 0.2 * np.random.rand(m, 1).astype(np.float32)
        self.max_val = 1

        # Generate the signals as a (m, n) array.
        self.y = np.zeros([self.num_series, window_size], dtype=np.float32)

        # Color of each vertex (TODO: make it more efficient by using a GLSL-based
        # color map and the index).
        if colors is None:
            color = np.repeat(
                np.random.uniform(size=(self.num_series, 3), low=0.5, high=0.9),
                window_size,
                axis=0,
            ).astype(np.float32)

        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        index = np.c_[
            np.repeat(np.repeat(np.arange(ts_cols), ts_rows), window_size),
            np.repeat(np.tile(np.arange(ts_rows), ts_cols), window_size),
            np.tile(np.arange(window_size), self.num_series),
        ].astype(np.float32)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program["a_position"] = self.y.reshape(-1, 1)
        self.program["a_color"] = color
        self.program["a_index"] = index
        self.program["u_scale"] = (1.0, 1.0)
        self.program["u_size"] = (ts_rows, ts_cols)
        self.program["u_n"] = self.window_size

        # TODO: remove?
        # gloo.set_state(
        #     clear_color="black",
        #     blend=True,
        #     blend_func=("src_alpha", "one_minus_src_alpha"),
        # )

        y2, x2 = self.figure.get_viewport_position(self.row, self.col + self.col_span)
        y1, x1 = self.figure.get_viewport_position(self.row + self.row_span, self.col)
        self.viewport_positions = (x1, y1, x2 - x1, y2 - y1)

        self.old_draw = self.figure.canvas.on_draw
        self.figure.canvas.on_draw = self.on_draw

    def on_draw(self, event):
        gloo.set_viewport(0, 0, *self.figure.canvas.physical_size)
        self.old_draw(event)
        gloo.set_viewport(*self.viewport_positions)
        self.program.draw("line_strip")

    def add_points(self, points, redraw=False):
        points = np.array(points).reshape(-1)
        self.max_val = max(self.max_val, np.max(np.abs(points)))
        self.y[:, :-1] = self.y[:, 1:]
        self.y[:, -1] = points
        self.program["a_position"].set_data(
            self.y.ravel().astype(np.float32) / self.max_val
        )
        if redraw:
            self.redraw()

    def update(self):
        app.process_events()


class ArrowPlot(Plot):
    """ LinePlot with arrow """

    def __init__(self, plot_options=None, **kwargs):
        super().__init__(**kwargs)

        plot_options = {} if plot_options is None else plot_options

        if "parent" in plot_options:
            self.view = plot_options["parent"].view
            plot_options["parent"] = self.view.scene

        plot_options.setdefault("parent", self.view.scene)
        plot_options.setdefault("width", 5)
        plot_options.setdefault("method", "gl")
        plot_options.setdefault("arrow_type", "stealth")

        """
        connect: "segment", "strip", or (_, 2) int32 array.
            ex. [[0, 1], [1, 2]] will connect vertices 0 and 1, vertices 1 and 2
        """

        plot_options.setdefault("connect", "segments")

        self.arrows = scene.visuals.Arrow(**plot_options)

    def update(self, lines, arrows, redraw=False):
        """ lines: array of shape (N, V, 2 or 3),
            N is the number of lines,
            V is the number of segments in each line
            2 or 3 depends on 2D or 3D

            arrows: array of shape (N, 4 or 6)
            last half is the centre of arrow,
            (last half) - (first half) is the arrow direction
        """
        if isinstance(self.view.camera, CustomPanZoomCamera):
            self.view.camera.expand_bounds(lines[:, :, 0].min(), lines[:, :, 1].min())
            self.view.camera.expand_bounds(lines[:, :, 0].max(), lines[:, :, 1].max())

        self.arrows.set_data(pos=lines, arrows=arrows)

        if redraw:
            self.redraw()


if __name__ == "__main__":
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
            "parent": ar1,
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
