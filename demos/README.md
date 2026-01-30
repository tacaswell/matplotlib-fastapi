# Matplotlib-FastAPI Demos

This directory contains example applications demonstrating how to use mpl-fastapi.

## Running the Demos

### Prerequisites

Install the package in development mode:

```bash
pip install -e .
# or with pixi:
pixi install
```

### Sine Wave Demo

The main demo shows interactive sine, cosine, and Lissajous curve plots:

```bash
uvicorn demos.sine_wave:app --reload
```

Then visit:
- http://localhost:8000/ - Redirects to plots list
- http://localhost:8000/plots - JSON API listing all available plots
- http://localhost:8000/plot/sine - Interactive sine wave viewer
- http://localhost:8000/plot/sine?frequency=2.0&amplitude=1.5 - With custom parameters
- http://localhost:8000/plot/cosine - Cosine wave with damping
- http://localhost:8000/plot/lissajous - Lissajous curves

## How It Works

1. **Define Parameters**: Create a Pydantic model for your plot parameters
2. **Create Generator**: Write a function that takes a `Figure` and parameters, populates the figure
3. **Register Plots**: Pass plot generators to `create_mpl_router()`
4. **Mount Router**: Include the router and static files in your FastAPI app

## Example

```python
from pydantic import BaseModel
from matplotlib.figure import Figure
from mpl_fastapi import create_mpl_router
import numpy as np

class MyPlotParams(BaseModel):
    n_points: int = 100

def my_plot_generator(fig: Figure, params: MyPlotParams) -> None:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 10, params.n_points)
    ax.plot(x, np.sin(x))
    ax.set_title(f"Plot with {params.n_points} points")

mpl = create_mpl_router({
    "myplot": (my_plot_generator, MyPlotParams, "My custom plot"),
})

app.include_router(mpl.router, prefix="/plots")
app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")
```

## Features Demonstrated

- **Parameter validation** using Pydantic
- **Multiple plots** in one router
- **No pyplot imports** - pure Figure API
- **Interactive controls** - zoom, pan, etc.
- **WebSocket-based** real-time updates
- **On-demand creation** - figures created when users connect
- **Automatic cleanup** - figures destroyed on disconnect
