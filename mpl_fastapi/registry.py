"""
Figure registry and backend management for matplotlib-fastapi.

This module provides a FigureRegistry class to manage matplotlib figures
and ensure they use the FastAPI backend. It does not import matplotlib.pyplot.
"""
from typing import Any
import weakref
import warnings
from collections import Counter

from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureCanvasBase as _FigureCanvasBase
from matplotlib.axes import Axes


class FigureRegistry:
    """
    A registry to wrap the creation of figures and track them.

    This instance will keep a hard reference to created Figures to ensure
    that they do not get garbage collected.

    Parameters
    ----------
    block : bool, optional
        Whether to wait for all figures to be closed before returning from
        show_all.

        If `True` block and run the GUI main loop until all figure windows
        are closed.

        If `False` ensure that all figure windows are displayed and return
        immediately.  In this case, you are responsible for ensuring
        that the event loop is running to have responsive figures.

        Defaults to True in non-interactive mode and to False in interactive
        mode (see `.is_interactive`).

    timeout : float, optional
        Default time to wait for all of the Figures to be closed if blocking.

        If 0 block forever.

    prefix : str, optional
        Prefix for auto-generated figure labels. Defaults to "Figure ".
    """

    def __init__(self, *, block: bool | None = None, timeout: float = 0, prefix: str = "Figure ") -> None:
        # settings stashed to set defaults on show
        self._timeout: float = timeout
        self._block: bool | None = block
        # the canonical location for storing the Figures this registry owns.
        # any additional views must never include a figure that is not a key but
        # may omit figures
        self._fig_to_number: dict[Figure, int] = dict()
        # Settings / state to control the default figure label
        self._prefix: str = prefix

    @property
    def figures(self) -> tuple[Figure, ...]:
        """Return tuple of all registered figures."""
        return tuple(self._fig_to_number)

    def _register_fig(self, fig: Figure) -> Figure:
        """Register a figure with this registry."""
        # if the user closes the figure by any other mechanism, drop our
        # reference to it.  This is important for getting a "pyplot" like user
        # experience
        def registry_cleanup(fig_wr: weakref.ref[Figure]) -> None:
            fig = fig_wr()
            if fig is not None:
                if fig.canvas is not None:
                    fig.canvas.mpl_disconnect(cid)
                self.close(fig)

        fig_wr = weakref.ref(fig)
        cid = fig.canvas.mpl_connect("close_event", lambda e: registry_cleanup(fig_wr))
        # Make sure we give the figure a quasi-unique label.  We will never set
        # the same label twice, but will not over-ride any user label (but
        # empty string) on a Figure so if they provide duplicate labels, change
        # the labels under us, or provide a label that will be shadowed in the
        # future it will be what it is.
        fignum = max(self._fig_to_number.values(), default=-1) + 1
        if fig.get_label() == "":
            fig.set_label(f"{self._prefix}{fignum:d}")
        self._fig_to_number[fig] = fignum
        return fig

    @property
    def by_label(self) -> dict[str, Figure]:
        """
        Return a dictionary of the current mapping labels -> figures.

        If there are duplicate labels, newer figures will take precedence.
        """
        mapping = {str(fig.get_label()): fig for fig in self.figures}
        if len(mapping) != len(self.figures):
            counts = Counter(str(fig.get_label()) for fig in self.figures)
            multiples = {k: v for k, v in counts.items() if v > 1}
            warnings.warn(
                (
                    f"There are repeated labels ({multiples!r}), but only the newest figure with that label can "
                    "be returned. "
                ),
                stacklevel=2,
            )
        return mapping

    @property
    def by_number(self) -> dict[int|str, Figure]:
        """
        Return a dictionary of the current mapping number -> figures.
        """
        return {fig.canvas.manager.num: fig for fig in self.figures if fig.canvas.manager is not None}

    def figure(self, *args: Any, **kwargs: Any) -> Figure:
        """
        Create a new figure and register it.

        Parameters are passed to matplotlib.figure.Figure constructor.
        """
        # Import here to avoid circular import
        from matplotlib.figure import Figure

        fig = Figure(*args, **kwargs)
        # Attach our custom canvas
        from mpl_fastapi.main import FastAPICanvas
        canvas = FastAPICanvas(fig)

        return self._register_fig(fig)

    def subplots(self, nrows: int = 1, ncols: int = 1, **kwargs: Any) -> tuple[Figure, Any]:
        """
        Create a figure and a set of subplots.

        This is similar to matplotlib.pyplot.subplots but returns
        a registered figure.
        """
        # Import here to avoid circular import
        from matplotlib.figure import Figure
        from mpl_fastapi.main import FastAPICanvas

        fig = Figure(**kwargs)
        canvas = FastAPICanvas(fig)

        # Create subplots
        axs = fig.subplots(nrows=nrows, ncols=ncols)

        return self._register_fig(fig), axs

    def subplot_mosaic(self, mosaic: str | list[list[str]], **kwargs: Any) -> tuple[Figure, dict[str, Axes]]:
        """
        Create a figure with a mosaic of named subplots.

        This is similar to matplotlib.pyplot.subplot_mosaic but returns
        a registered figure.
        """
        # Import here to avoid circular import
        from matplotlib.figure import Figure
        from mpl_fastapi.main import FastAPICanvas

        fig = Figure(**kwargs)
        canvas = FastAPICanvas(fig)

        # Create subplot mosaic
        axd = fig.subplot_mosaic(mosaic)  # type: ignore[arg-type]

        return self._register_fig(fig), axd

    def close_all(self) -> None:
        """
        Close all Figures known to this Registry.

        This will do four things:

        1. call the ``.destroy()`` method on the manager
        2. clears the Figure on the canvas instance
        3. replace the canvas on each Figure with a new FigureCanvasBase instance
        4. drops its hard reference to the Figure

        If the user still holds a reference to the Figure it can be revived by
        passing it to display.
        """
        for fig in list(self.figures):
            self.close(fig)

    def close(self, val: str | int | Figure) -> None:
        """
        Close (meaning destroy the UI) and forget a managed Figure.

        This will do two things:

        - start the destruction process of any UI (the event loop may need to
          run to complete this process and if the user is holding hard
          references to any of the UI elements they may remain alive).
        - Remove the Figure from this Registry.

        We will no longer have any hard references to the Figure, but if
        the user does the Figure (and its components) will not be garbage
        collected.  Due to the circular references in Matplotlib these
        objects may not be collected until the full cyclic garbage collection
        runs.

        If the user still has a reference to the Figure they can re-show the
        figure, but the FigureRegistry will not be aware of it.

        Parameters
        ----------
        val : 'all' or int or str or Figure

            - The special case of 'all' closes all open Figures
            - If any other string is passed, it is interpreted as a key in
              `by_label` and that Figure is closed
            - If an integer it is interpreted as a key in `.by_number` and that
              Figure is closed
            - If it is a Figure instance, then that figure is closed
        """
        if val == "all":
            self.close_all()
            return
        # or do we want to close _all_ of the figures with a given label / number?
        if isinstance(val, str):
            fig = self.by_label[val]
        elif isinstance(val, int):
            fig = self.by_number[val]
        else:
            fig = val
            if fig not in self.figures:
                raise ValueError(
                    "Trying to close a figure not associated with this Registry."
                )
        if fig.canvas.manager is not None:
            fig.canvas.manager.destroy()
            # disconnect figure from canvas
            fig.canvas.figure = None  # type: ignore[assignment]
            # disconnect canvas from figure
            _FigureCanvasBase(figure=fig)
        assert fig.canvas.manager is None
        self._fig_to_number.pop(fig, None)
        return


def select_gui_toolkit(backend_class: type) -> None:
    """
    Set the matplotlib backend to use our custom FastAPI backend.

    This configures matplotlib to use the provided backend class
    without importing pyplot.

    Parameters
    ----------
    backend_class : class
        The backend class (should have FigureCanvas and FigureManager).
    """
    import matplotlib
    import matplotlib.backend_bases

    # Register our backend module
    matplotlib.backend_bases.register_backend('module://mpl_fastapi.backend', backend_class)

    # Use our backend
    matplotlib.use('module://mpl_fastapi.backend', force=True)
