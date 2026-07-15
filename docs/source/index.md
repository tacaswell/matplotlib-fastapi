# mpl-fastapi documentation


```{toctree}
---
hidden: true
---

reference.md
explain.md
tutorials.md
how-to.md

```


The goal of this project is to provide a
[fastapi](https://fastapi.tiangolo.com)
[`APIRouter`](https://fastapi.tiangolo.com/reference/apirouter/?h=apirouter)
that serves interactive [Matplotlib](https://matplotlib.org) figures via a
websocket.  This provides the ability to serve visualization developed in
Python via Matplotlib to the web with server-side rendering and all user
events.


```{admonition} 🤖 generated code ahead

This project was half about generating a useful tool and half about having a
real project to try out LLM tools on.

The initial proof of concept was done in late 2021/early 2022 and then sat
somewhere too far down my todo list to ever get done until early 2026.

I am not sure about the future of LLMs in this project.  If this moves beyond
being a toy we will move to follow
[Matplotlib's
policy](https://matplotlib.org/devdocs/devel/contribute.html#use-of-generative-ai).

```

## Overview

This project provides a Configurable Application for serving pre-defined
interactive Matplotlib figures via FastAPI.  A FastAPI Router is also exposed
for embedding into other servers.  The router provides a handful of static
routes and then a web-socket route per defined figure.  When a client connects
the websocket is used to serve a fully interactive figure.  All of the
rendering is done on the server side sent to the client as a png.  This means
that the raw data does not need to be shipped to the client.  User interaction
events are forwarded from the client to the server to be handled.  This means
that any interactions written Matplotlib's [event
system](https://matplotlib.org/stable/users/explain/figure/event_handling.html#id1)
will directly work with no modification.  However, doing the rendering and
event handling on the server side will introduce additional latency.

The server provides the hooks to introduce AuthN and AuthZ as well as optionally
inject identity information into the plot initialization function.

The package contains a TypeScript client and three Python clients (headless, tk
and Qt) understand the http and websocket protocols that the server provides.
In addition, in both languages the low-level client classes are exposed for
custom applications.

## Minimal example

To define a figure to be served you must provide two things:

1. a `pydandtic.BaseModel` to describe the parameters
2. a function that given an empty figure and the BaseModel, construct the
   visualization

The `BaseModel` is required to define the parameter parsing by FastAPI so it is
simplest to have the user directly provide it.  There is the added benefit that
the user can provide additional validation and a description which can then be
used to build input forms[^1].

The expected signature of the function is:

```python

class Params(BaseModel):
    ...

def plot_init(fig: Figure, params: Params) -> Any:
    ...

```

The actual name of the class and function do not matter.  If using the default
configurable application you will need to explicitly export them.

Concretely:

```python
# first_plot.py
from pydantic import BaseModel
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from mpl_fastapi import PlotConfig, InitConfig


class InitParams(BaseModel):
    """Parameters for sine wave visualization."""

    frequency: float = 3.5


def create_plot(fig: Figure, params: InitParams) -> None:
    ax = fig.subplots()
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\sin(\omega\theta)$")

    ω = params.frequency
    θ = np.linspace(0, 2 * np.pi, 256)

    (ln,) = ax.plot(θ, np.sin(ω * θ), label=rf"$\omega={ω:.2f}$")
    ax.legend()


# expose your plot to the server
plots: dict[str, InitConfig] = {
    "sine": PlotConfig(
        "Sine wave",
        InitConfig(
            create_plot,
            InitParams,
        ),
    )
}

```

If we then invoke the bundled application via:

```sh
python -m mpl_fastapi first_plot:plots --port 8080
```

will serve the plot on `localhost` (be default with a single-user key).  The
first argument is `module:attribute` of where a `dict[str, InfoConfig]` can be
found.  Both names are arbitrary so long as the module can be imported.  All of
the standard `uvicorn` command line flags can be passed and are forwarded
through to `uvicorn`.

Connecting to the displayed url with a browser will show a live figure.  Each
connection to the server is a new websocket which in turn generates it's own
`Figure` instance.  Thus, multiple independent clients can be served
simultaneously and will not conflict with each other.  All of the user events
from the browser (mouse motion, mouse buttons, key press/release, and
enter/exit events) are sent back to the server over the websocket and handled
on the server side.  Thus, any code that currently uses Matplotlib's event
system will "just work" out of the box.

By providing an additional function


## Development

This project uses [pixi](https://pixi.prefix.dev/latest/) for dependency,
package management, and development tooling.  The three main entry points are:

```bash
# to run example server with hot-reloading of Python and TS
pixi run dev
# serve docs with auto-rebuild
pixi run docs-serve
# run all of the linting
pixi run check
```


[^1]: In the future it is conceivable that we could auto-construct the
    BaseModel from introspecting the signature, however that FastAPI does not
    do this so it is unlikely to be a good idea.
