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
uvicorn demos.demo_server:app --reload
```

Then visit:
- http://localhost:8000/plots - HTML plots list
- http://localhost:8000/plots/plots - JSON API listing all available plots
- http://localhost:8000/plots/plot/sine - Interactive sine wave viewer
- http://localhost:8000/plots/plot/sine?frequency=2.0&amplitude=1.5 - With custom parameters
- http://localhost:8000/plots/plot/cosine - Cosine wave with damping
- http://localhost:8000/plots/plot/lissajous - Lissajous curves
- **http://localhost:8000/embeddable** - Embeddable component demos
- **http://localhost:8000/react-app/** - React integration example

### Embeddable Component Demo

The `embeddable_demo.html` file demonstrates the new **MatplotlibEmbeddable** JavaScript API for framework-agnostic plot integration. This shows:

- Basic usage with auto-connection
- Manual connection control
- Programmatic parameter updates
- Multiple independent plots on one page
- Animation examples
- Error handling
- React integration patterns

Visit http://localhost:8000/embeddable to see interactive examples.

### React Integration Example

The `react-example/` directory contains a complete React application demonstrating
how to use the `mpl-fastapi` npm package:

```bash
# First, build the npm package (from repo root)
npm run build:npm

# Install and build React app
cd demos/react-example
npm install
npm run build

# Run FastAPI server (from repo root)
uvicorn demos.demo_server:app --reload
```

Visit http://localhost:8000/react-app/ - everything is served through FastAPI, no separate dev server needed!

## Testing the npm Package Locally

The TypeScript client can be published to npm as `mpl-fastapi`. For testing locally before publishing:

### Option 1: npm link (recommended for development)

```bash
# Build the npm package
npm run build:npm

# Create a global symlink
npm link

# In your test project
cd /path/to/your/project
npm link mpl-fastapi
```

Then import in your project:
```javascript
import { MatplotlibEmbeddable } from 'mpl-fastapi';
```

### Option 2: file: dependency

In your test project's `package.json`:
```json
{
  "dependencies": {
    "mpl-fastapi": "file:/path/to/matplotlib-fastapi"
  }
}
```

### Option 3: ES Module import from dist

After building (`npm run build:npm`), you can import directly in modern browsers:
```html
<script type="module">
  import { MatplotlibEmbeddable } from '/path/to/dist/mpl-fastapi.js';
</script>
```

### Option 4: Use importmap for cleaner imports

```html
<script type="importmap">
{
  "imports": {
    "mpl-fastapi": "/path/to/dist/mpl-fastapi.js"
  }
}
</script>
<script type="module">
  import { MatplotlibEmbeddable } from 'mpl-fastapi';
</script>
```

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
