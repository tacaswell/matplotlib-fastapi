# matplotlib-fastapi


**Interactive, WebSocket-based matplotlib plotting for FastAPI applications**

matplotlib-fastapi is a library that integrates matplotlib with FastAPI to
provide real-time, interactive plotting capabilities. It allows you to create
mountable FastAPI routers that serve interactive matplotlib figures with pan,
zoom, and dynamic parameter updates—all without page reloads.

## Quick Start

### Minimal Example

Here's a minimal example of adding an interactive matplotlib plot to an
existing FastAPI application:

```python
from fastapi import FastAPI
from matplotlib.figure import Figure
from pydantic import BaseModel, Field
import numpy as np

from mpl_fastapi import create_mpl_router, InitConfig, PlotConfig

# Define parameter model for your plot
class SinePlotParams(BaseModel):
    frequency: float = Field(default=1.0, ge=0.1, le=10.0)
    amplitude: float = Field(default=1.0, ge=0.1, le=5.0)

# Define plot generator function
def create_sine_plot(fig: Figure, params: SinePlotParams) -> dict:
    """Create a sine wave plot and return state for updates."""
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4 * np.pi, 200)
    y = params.amplitude * np.sin(params.frequency * x)
    line, = ax.plot(x, y)
    ax.set_title(f"Sine Wave: A={params.amplitude}, f={params.frequency}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return {"line": line, "x": x, "params": params}

# Create your FastAPI app
app = FastAPI()

# Create matplotlib router with plot configuration
mpl = create_mpl_router({
    "sine": PlotConfig(
        description="Interactive sine wave plot",
        init=InitConfig(
            function=create_sine_plot,
            params_model=SinePlotParams,
        ),
    ),
})

# Mount the router and static files
app.include_router(mpl.router, prefix="/plots")
app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")

# Run with: uvicorn app:app --reload
# Visit: http://localhost:8000/plots
```

### Adding Dynamic Updates

To make your plots dynamically updatable, add an update function:

```python
from mpl_fastapi import UpdateConfig

class SineUpdateParams(BaseModel):
    phase: float = Field(default=0.0, ge=0.0, le=6.28)

def update_sine_plot(state: dict, params: SineUpdateParams) -> dict:
    """Update the plot using cached state."""
    y = state["params"].amplitude * np.sin(
        state["params"].frequency * state["x"] + params.phase
    )
    state["line"].set_ydata(y)
    return state

mpl = create_mpl_router({
    "sine": PlotConfig(
        description="Interactive sine wave with phase control",
        init=InitConfig(
            function=create_sine_plot,
            params_model=SinePlotParams,
        ),
        update=UpdateConfig(
            function=update_sine_plot,
            params_model=SineUpdateParams,
        ),
    ),
})
```

Now users can adjust the phase parameter in real-time without reconnecting!

## JavaScript Client

You can embed matplotlib plots into React, Vue, or any JavaScript framework
using the client component:

```html
<script src="/plots/component.js"></script>
<div id="my-plot"></div>

<script>
  const plot = new MatplotlibClient({
    container: document.getElementById('my-plot'),
    plotName: 'sine',
    baseUrl: '/plots',
    initParams: { frequency: 2.0, amplitude: 1.5 },
    updateParams: { phase: 0.0 },
    onUpdate: (params) => console.log('Updated:', params),
    autoConnect: true
  });

  // Programmatically update
  plot.update({ phase: 1.57 });
</script>
```

See `demos/client_demo.html` for comprehensive examples.


## Development

### Setup

```bash
# Clone and install with pixi (manages Python + Node.js)
git clone https://github.com/tacaswell/matplotlib-fastapi.git
cd matplotlib-fastapi
pixi install

```

### Development Workflow

```bash
# Start development server with hot reload
pixi run dev

# Run tests
pixi run pytest

# Linting and type checking
pixi run check  # Run all checks (Python + TypeScript)
```

## Examples

See the `demos/` directory for complete examples:

- `demo_server.py`: Server with basic
- `client_demo.html`: Examples of JavaScript client usage
- `index.html`: Template-based integration example
- `react-example`: A

## License

3-clause BSD License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.rst](CONTRIBUTING.rst) for guidelines.

## Credits

Created by Thomas A Caswell and contributors. See [AUTHORS.rst](AUTHORS.rst) for the full list.
