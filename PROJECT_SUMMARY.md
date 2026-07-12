# matplotlib-fastapi Project Summary

## Project Overview

**matplotlib-fastapi** is a library that integrates matplotlib with FastAPI to provide interactive, WebSocket-based plotting capabilities. It allows developers to create mountable FastAPI routers that serve interactive matplotlib figures with real-time updates.

### Key Features
- **WebSocket-based interactivity**: Pan, zoom, and interact with plots in real-time
- **Mountable router pattern**: Easy integration into existing FastAPI applications
- **Plot generator functions**: Define plots as functions that take parameters
- **Dynamic plot updates**: Update plots without reconnection using state caching
- **Parameter validation**: Pydantic models for type-safe parameter handling
- **Differential rendering**: Efficient image updates (only changed pixels sent)
- **No pyplot dependency**: Pure Figure-based API for better control
- **JavaScript client**: Framework-agnostic JavaScript API for composable plot integration

## Architecture

### Three-Layer Design

1. **Backend Layer** (`mpl_fastapi/mpl_backend.py`)
   - `FastAPICanvas`: Custom matplotlib canvas with WebSocket communication
   - `NavigationToolbar2FastAPI`: Interactive toolbar (pan, zoom, home, etc.)
   - `FastAPIManger`: Figure manager for lifecycle management
   - `FastAPIBackend`: Matplotlib backend registration
   - Handles rendering, events, and differential image updates

2. **Router Layer** (`mpl_fastapi/router.py`)
   - `create_mpl_router()`: Factory function for creating mountable routers
   - HTTP endpoints: `/` (HTML list), `/plots` (JSON API), `/plot/{name}` (viewer)
   - WebSocket endpoint: `/ws/v0/{name}` (interactive connection)
   - Component endpoints: `/component.js` (client bundle), `/api/plots/{name}/schema` (JSON schema)
   - Configuration structures: `PlotConfig`, `InitConfig`, `UpdateConfig`
   - Response models: `PlotInfo`, `PlotsListResponse`
   - Type aliases: `PlotGenerator`, `UpdateFunction`

3. **Public API** (`mpl_fastapi/__init__.py`)
   - Exports: `create_mpl_router`, `MPLRouter`, `PlotGenerator`, `UpdateFunction`
   - Configuration: `PlotConfig`, `InitConfig`, `UpdateConfig`
   - Response models: `PlotInfo`, `PlotsListResponse`
   - Clean interface for external use

### Key Design Patterns

**PlotConfig Structure**: Plots are configured using structured dataclasses:
```python
PlotConfig(
    description="Human-readable description",
    init=InitConfig(
        function=plot_generator,  # Callable[[Figure, Any], Any]
        params_model=InitParamsModel
    ),
    update=UpdateConfig(  # Optional
        function=update_function,  # Callable[[Any, Any], Any]
        params_model=UpdateParamsModel
    )
)
```

**Plot Generator Pattern**: Generator functions return opaque state for updates:
```python
def plot_generator(fig: Figure, params: BaseModel) -> dict:
    """Populate the figure with plot elements and return state."""
    ax = fig.add_subplot(111)
    line, = ax.plot(x, y)
    return {"line": line, "ax": ax, "params": params}
```

**Update Function Pattern**: Update functions receive and return state:
```python
def update_function(state: dict, params: BaseModel) -> dict:
    """Update plot using cached state, return updated state."""
    state["line"].set_ydata(new_y)
    return state
```

**On-Demand Figure Creation**: Figures are created when a WebSocket connection is established and destroyed when it disconnects. State is cached per-connection.

**Dependency Injection**: FastAPI dependencies used for validation (e.g., `validate_plot_name`)

## File Structure

```
mpl_fastapi/
├── __init__.py           # Public API exports
├── mpl_backend.py        # Matplotlib backend implementation
├── router.py             # FastAPI router factory
├── utils.py              # Utility functions
├── static/               # Static assets (CSS, JS, images)
│   ├── css/              # Stylesheets for web interface
│   ├── js/
│   │   ├── src/          # TypeScript source files
│   │   │   ├── index.ts           # Main entry point
│   │   │   ├── types.ts           # Type definitions
│   │   │   ├── websocket-manager.ts  # WebSocket lifecycle
│   │   │   ├── figure.ts          # Core figure renderer
│   │   │   └── client.ts          # JavaScript client component
│   │   └── dist/         # Compiled JavaScript (build artifact)
│   │       └── component.js       # Production bundle
│   └── images/           # Toolbar icons
├── templates/            # Jinja2 templates
│   ├── plots_list.html   # HTML listing with parameter forms
│   └── figure.html       # Plot viewer template
└── tests/                # Test suite

demos/
├── sine_wave.py          # Example FastAPI application
├── client_demo.html      # JavaScript client demos
└── README.md             # Demo documentation

# Build system
build_backend.py          # Custom build backend integrating TypeScript
pyproject.toml            # Python packaging configuration
package.json              # Node.js/TypeScript dependencies
tsconfig.json             # TypeScript compiler configuration
esbuild.config.mjs        # JavaScript bundler configuration
pixi.toml                 # Environment management (Python + Node.js)
```

## Modern Infrastructure

### Packaging & Environment
- **pyproject.toml**: Modern build system with custom build backend
- **setuptools-scm**: Dynamic versioning from git tags
- **Custom build backend**: Integrates TypeScript compilation into Python packaging
- **pixi.toml**: Unified environment management for Python 3.12+ and Node.js 20+
- **Python dependencies**: matplotlib >= 3.10.0, FastAPI >= 0.115.0, pydantic >= 2.0
- **JavaScript tooling**: TypeScript 5.3+, esbuild for fast compilation
- **Type annotations**: Comprehensive typing throughout, `py.typed` marker for Python

### Code Quality
- **ruff**: Linting and formatting for Python
- **mypy**: Static type checking for Python
- **TypeScript strict mode**: Full type safety in JavaScript components
- **Structured logging**: Uses Python's logging module (no print statements)
- **FastAPI best practices**: HTTPException, Pydantic response models, dependencies

### Build System Architecture

The project uses a **Python-led build system** where Python's setuptools orchestrates the entire build process:

1. **Custom Build Backend** (`build_backend.py`):
   - Wraps `setuptools.build_meta`
   - Automatically runs `npm install` and `npm run build` before Python packaging
   - Validates Node.js availability
   - Ensures TypeScript is compiled before wheel/sdist creation
   - Provides graceful error messages if Node.js is missing

2. **TypeScript Compilation** (esbuild):
   - Compiles `src/*.ts` → `dist/component.js` (bundled IIFE)
   - Strict mode enabled, ES2020 target
   - Source maps for debugging
   - Version injection from `package.json`

3. **Development Workflow**:
   - `pixi run dev`: Runs TypeScript watch mode and Python server in parallel
   - Hot reload: TypeScript recompiles automatically, browser auto-refreshes
   - `pixi run build`: Full production build (TypeScript + Python)

4. **Install-time Build**:
   - When users run `pip install`, custom backend automatically builds TypeScript
   - No manual build step required for end users
   - Fails clearly if Node.js is unavailable

## Project Evolution

### Phase 1-5: Core Architecture ✅
- Migrated from versioneer to setuptools-scm
- Converted from standalone app to mountable router pattern
- Implemented plot generator pattern with state caching
- Added Pydantic validation and FastAPI best practices
- Created dynamic update API with WebSocket `update_params` messages
- Comprehensive type annotations throughout

### Phase 6: JavaScript Client Component ✅
- Created `MatplotlibClient` class for framework-agnostic integration
- Programmatic API supporting React, Vue, Angular, vanilla JS
- Added `/component.js` and `/api/plots/{name}/schema` endpoints
- Auto-generated update forms from JSON schemas
- Full lifecycle control with event callbacks

### Phase 7: Download Functionality ✅
- Connection-scoped figure cache with UUID-based access
- HTTP download endpoint supporting all matplotlib formats
- WebSocket protocol extension for `connection_id` delivery
- Configurable DPI, transparency, and format options

### Phase 8: WebSocket Refactoring ✅
- **WebSocketManager pattern**: Centralized lifecycle management
- Event registration before connection to eliminate race conditions
- Consistent behavior across template and client components
- Clean separation: WebSocket management vs figure rendering
- Handler registration API: `onOpen()`, `onMessage()`, `onClose()`, `onError()`

### Phase 9: TypeScript Migration ✅

**Motivation**: Eliminate type shims, improve maintainability, enable modern JavaScript tooling

**Key Design Decisions**:
1. **Unified Component Architecture (Option C)**:
   - Single `MatplotlibClient` class replacing dual `mpl.js`/`mpl_embeddable.js` approach
   - Supports both template-based and programmatic usage patterns
   - Eliminates code duplication and maintenance burden
   - Clean API: `new MatplotlibClient(config)` for all use cases

2. **Build-at-Install-Time**:
   - Custom Python build backend (`build_backend.py`) wraps setuptools
   - Automatically runs `npm install` and `npm run build` during packaging
   - End users get compiled JavaScript without manual build steps
   - Fails gracefully if Node.js unavailable with clear error message

3. **Python-Led Build System**:
   - Python's setuptools orchestrates entire build process
   - TypeScript compilation integrated into Python packaging workflow
   - Single source of truth: `pyproject.toml` with `build-backend = "build_backend"`
   - Version synchronized via setuptools-scm and package.json

4. **Developer Experience**:
   - `pixi run dev`: Parallel TypeScript watch + Python server with hot reload
   - TypeScript strict mode with comprehensive type definitions
   - esbuild for fast compilation (milliseconds vs seconds)
   - Source maps for debugging production code

**Implementation**:
- **TypeScript Source** (`mpl_fastapi/static/js/src/`):
  - `types.ts`: 300+ lines of complete type definitions (ClientConfig, ServerMessage, ClientMessage, JSONSchema)
  - `websocket-manager.ts`: WebSocket lifecycle management as ES6 class
  - `figure.ts`: Core figure renderer (~700 lines) with all matplotlib interactions
  - `client.ts`: Unified component with CSS injection, schema fetching, form generation
  - `index.ts`: Main entry point exposing `window.mpl` namespace

- **Build Configuration**:
  - `tsconfig.json`: Strict mode, ES2020 target, source maps, declaration files
  - `esbuild.config.mjs`: IIFE bundle, minification, version injection, watch mode
  - `package.json`: TypeScript 5.3+, esbuild 0.20+, dev scripts
  - `pixi.toml`: nodejs dependency, js-install/js-build/js-watch/dev tasks

- **Python Integration**:
  - `mpl_backend.py`: Reads `dist/component.js` instead of legacy files
  - `router.py`: Serves compiled bundle, adds source map endpoint
  - Templates updated: `new mpl.Figure()` (capitalized, idiomatic)
  - Legacy `mpl.js` and `mpl_embeddable.js` removed

**Benefits**:
- ✅ Full type safety across Python and JavaScript
- ✅ Eliminates type shims and "any" types
- ✅ Single component supporting all use cases
- ✅ Modern development workflow with hot reload
- ✅ Easier to maintain and extend
- ✅ Better IDE support and autocomplete
- ✅ Automated build process for end users
- ✅ Production-ready minified bundles

## API Usage Example

```python
from fastapi import FastAPI
from matplotlib.figure import Figure
from pydantic import BaseModel, Field
import numpy as np

from mpl_fastapi import InitConfig, PlotConfig, UpdateConfig, create_mpl_router

# Define parameter models with validation
class SinePlotParams(BaseModel):
    frequency: float = Field(default=1.0, ge=0.1, le=10.0)
    amplitude: float = Field(default=1.0, ge=0.1, le=5.0)

class SineUpdateParams(BaseModel):
    phase: float = Field(default=0.0, ge=0.0, le=6.28, description="Phase shift")

# Define plot generator function (returns state)
def create_sine_plot(fig: Figure, params: SinePlotParams) -> dict:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4*np.pi, 200)
    y = params.amplitude * np.sin(params.frequency * x)
    line, = ax.plot(x, y)
    ax.set_title(f"Sine: A={params.amplitude}, f={params.frequency}")
    return {"line": line, "x": x, "params": params}

# Define update function (modifies state)
def update_sine_plot(state: dict, params: SineUpdateParams) -> dict:
    y = state["params"].amplitude * np.sin(
        state["params"].frequency * state["x"] + params.phase
    )
    state["line"].set_ydata(y)
    return state

# Create router with PlotConfig
app = FastAPI()
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

# Mount router and static files
app.include_router(mpl.router, prefix="/plots")
app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")

# Run: uvicorn app:app --reload
# Visit: http://localhost:8000/plots
```

## JavaScript Client API

The JavaScript client provides a framework-agnostic JavaScript API for integrating matplotlib figures into any web application without requiring server-rendered HTML templates.

### Key Features
- **Composable**: Can be embedded into React, Vue, Angular, or vanilla JavaScript applications
- **Programmatic instantiation**: No need for predefined HTML templates
- **Full lifecycle control**: Manual connect/disconnect with event callbacks
- **Auto-generated forms**: Dynamically creates update forms from JSON schema
- **Multiple plots**: Support for multiple independent plots on the same page
- **Clean API**: Simple JavaScript class with intuitive methods
- **High-DPI support**: Automatically sends device pixel ratio for crisp rendering on retina displays
- **Error handling**: Proper WebSocket error handling with cleanup and callbacks
- **Connection state tracking**: Polls WebSocket readyState without interfering with internal handlers

### JavaScript Usage

```html
<!-- Include the component bundle -->
<script src="/plots/component.js"></script>

<div id="my-plot"></div>

<script>
  // Create and configure the plot
  const plot = new MatplotlibClient({
    container: document.getElementById('my-plot'),
    plotName: 'sine',
    baseUrl: '/plots',
    initParams: { frequency: 2.0, amplitude: 1.0 },
    updateParams: { phase: 0.0 },
    onConnect: () => console.log('Connected!'),
    onError: (error) => console.error('Error:', error),
    onUpdate: (params) => console.log('Updated:', params),
    onDisconnect: () => console.log('Disconnected'),
    showToolbar: true,       // Show matplotlib toolbar (default: true)
    showUpdateForm: true,    // Show auto-generated update form (default: true)
    autoConnect: true        // Auto-connect on creation (default: true)
  });

  // Programmatically update parameters
  plot.update({ phase: 1.57 });

  // Check connection status
  if (plot.isConnected()) {
    console.log('Currently connected');
  }

  // Cleanup when done
  plot.disconnect();
</script>
```

### Constructor Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `container` | HTMLElement | **required** | DOM element to render the plot into |
| `plotName` | string | **required** | Name of the plot to load |
| `baseUrl` | string | `''` | Base URL for the API endpoints |
| `initParams` | object | `{}` | Initial parameters for plot generation |
| `updateParams` | object | `{}` | Initial update parameters |
| `onConnect` | function | `() => {}` | Callback when WebSocket connects |
| `onError` | function(error) | console.error | Callback for errors |
| `onUpdate` | function(params) | `() => {}` | Callback after parameters update |
| `onDisconnect` | function | `() => {}` | Callback when disconnected |
| `showToolbar` | boolean | `true` | Show matplotlib toolbar |
| `showUpdateForm` | boolean | `true` | Show auto-generated update form |
| `staticPath` | string | `'/mpl-static'` | Path to static assets |
| `autoConnect` | boolean | `true` | Automatically connect on creation |

### Methods

- **`connect()`**: Connect to the WebSocket and initialize the plot
- **`disconnect()`**: Close connection and cleanup resources
- **`update(params)`**: Update plot parameters (e.g., `plot.update({ phase: 1.5 })`)
- **`isConnected()`**: Returns `true` if currently connected
- **`getUpdateParams()`**: Get current update parameters
- **`getInitParams()`**: Get initial parameters

### React Integration Example

```jsx
import { useEffect, useRef } from 'react';

function MatplotlibPlot({ plotName, initParams, onUpdate }) {
  const containerRef = useRef(null);
  const plotRef = useRef(null);

  useEffect(() => {
    // Create plot on mount
    plotRef.current = new MatplotlibClient({
      container: containerRef.current,
      plotName: plotName,
      baseUrl: '/plots',
      initParams: initParams,
      onUpdate: onUpdate,
      showUpdateForm: false  // Use React state instead
    });

    // Cleanup on unmount
    return () => {
      if (plotRef.current) {
        plotRef.current.disconnect();
      }
    };
  }, [plotName]);

  const handleUpdate = (params) => {
    if (plotRef.current) {
      plotRef.current.update(params);
    }
  };

  return (
    <div>
      <div ref={containerRef} />
      <button onClick={() => handleUpdate({ phase: Math.random() * 6.28 })}>
        Random Phase
      </button>
    </div>
  );
}
```

### New API Endpoints

1. **`GET /plots/component.js`** - Serves the TypeScript-compiled component bundle
   - Single IIFE bundle with all dependencies
   - Includes toolbar configuration (items, extensions, default format)
   - Exposes `window.mpl` and `window.MatplotlibClient`

2. **`GET /plots/component.js.map`** - Serves source map for debugging
   - Enables TypeScript debugging in browser DevTools
   - Maps compiled JavaScript back to original TypeScript source

3. **`GET /plots/api/plots/{plot_name}/schema`** - Returns plot schemas as JSON
   - Returns `init_schema`, `update_schema`, and description
   - Used by the component to auto-generate update forms
   - Example response:
     ```json
     {
       "plot_name": "sine",
       "description": "Interactive sine wave",
       "init_schema": { "properties": {...}, "required": [...] },
       "update_schema": { "properties": {...}, "required": [...] }
     }
     ```

### Demo

See `demos/client_demo.html` for comprehensive examples including:
- Basic usage with auto-connection
- Manual connection control
- Multiple plots on one page
- Custom parameter updates
- Animation examples
- Error handling

## Key Technical Details

### WebSocket Protocol
 1. Client connects to `/{prefix}/ws/v0/{plot_name}?param1=value1&param2=value2`
    (the `v0` segment is the protocol version)
 2. Server validates plot exists BEFORE accepting connection (rejects with code 1008 if invalid)
 3. Server validates the request `Origin` against the allow-list and accepts the WebSocket connection
 4. Client (via `WebSocketManager`) sends a single consolidated `init` message as the
    REQUIRED first message:
    - `protocol_version` - client protocol version
    - `device_pixel_ratio` - device pixel ratio for high-DPI rendering
    - `supports_binary` - binary message support flag (always `true` in the reference client)
 5. Server generates a unique UUID connection ID
 6. Server validates query parameters (and any `_update.*` params) using the Pydantic model(s)
 7. Server creates Figure and calls the generator function
 8. Generator returns an opaque state object (cached for the connection lifetime)
 9. Server attaches FastAPICanvas and stores (Figure, Canvas) in cache with the connection ID
 10. Server replies with a single consolidated `config` message containing:
    - `protocol_version`, `connection_id`
    - `figure` (size, dpi, label)
    - `toolbar` (items, history state)
    - `save` (formats, default_format)
    - `image` (format), `update_schema`, `init_params`, `update_params`
 11. Client applies the config (builds toolbar, resizes canvas) and sends `refresh`;
     server responds with the initial image
12. Client sends events (mouse, toolbar buttons, update requests, save requests)
13. For update requests:
    - Server validates update parameters
    - Calls update function with cached state
    - Update function modifies plot and returns new state
    - Server triggers redraw via `canvas.draw_idle()`
14. For save requests:
    - Client uses stored connection_id to construct download URL
    - HTTP GET to `/download/{connection_id}?format=png` (separate from WebSocket)
    - Server retrieves figure from cache using connection_id
    - Server calls matplotlib's `savefig()` with requested format
    - Returns file as streaming download
15. Server processes events and sends updates (full or diff images)
16. Connection closed → figure and state destroyed, removed from cache

### WebSocketManager Architecture
- **Purpose**: Centralized WebSocket lifecycle management
- **Implementation**: TypeScript class in `websocket-manager.ts`
- **Responsibilities**:
  - Creates and manages WebSocket connection
  - Provides event registration before connection
  - Ensures proper message ordering
  - Handles connection state tracking
  - Cleanup on disconnect
- **Usage Pattern**:
  ```javascript
  // Create manager with URL
  const manager = new mpl.WebSocketManager(ws_url);

  // Register handlers BEFORE connecting
  manager.onOpen(() => { /* handle open */ });
  manager.onMessage((event) => { /* handle message */ });
  manager.onClose((event) => { /* handle close */ });
  manager.onError((event) => { /* handle error */ });

  // Connect after all handlers registered
  manager.connect();
  ```
- **Integration**:
  - `mpl.Figure` accepts WebSocketManager or URL string
  - Template passes URL, figure creates manager internally
  - Client component creates manager, passes to figure
  - Both approaches guarantee handlers registered before connection

### Download Functionality
- **Connection-scoped cache**: `_active_figures: dict[str, tuple[Figure, FastAPICanvas]]`
- **Lifecycle**: Cache entry created on WebSocket accept, removed on disconnect
- **Endpoint**: `GET /download/{connection_id}`
- **Query parameters**:
  - `format`: Output format (png, pdf, svg, eps, ps, jpg, jpeg, tiff, tif)
  - `dpi`: DPI for raster formats (default: 100)
  - `transparent`: Transparency for PNG (default: false)
- **Response**: StreamingResponse with proper MIME type and filename
- **Filename**: Automatically uses figure ID (e.g., `sine.png`)
- **Error handling**:
  - 404 if connection_id not found (WebSocket closed)
  - 400 if unsupported format
  - 500 if savefig fails
- **Format support**: All matplotlib-supported formats via `fig.savefig()`

### Differential Rendering
- Uses numpy to compute pixel differences between frames
- Sends only changed pixels for efficiency
- Falls back to full image if transparency detected
- PNG encoding via PIL

### Event Handling
Event handlers in FastAPICanvas follow the pattern:
```python
async def handle_{event_type}(self, ev: dict[str, Any], websocket: WebSocket) -> None:
    # Process event and queue responses
    # Some handlers have unused params (required by callback API)
```

### Update Message Protocol
New WebSocket message type for dynamic updates:
```javascript
// Client sends update request
{
    "type": "update_params",
    "params": {
        "phase": 1.57,
        // ... other update parameters
    }
}

// Server validates, calls update function, and triggers redraw
// Client receives standard image update messages
```

## Configuration & Settings

### Static File Mounting
The router requires separate mounting of static files:
```python
app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")
```
Default path: `/mpl-static`

### Template Customization
Override default templates:
```python
mpl = create_mpl_router(
    plot_generators,
    template_dir="/path/to/custom/templates"
)
```

### Current Implementation Notes
- **Coordinate system**: Uses viewport-relative coordinates (`clientX/clientY`) for accurate positioning
- **High-DPI rendering**: Client sends logical pixel dimensions and device pixel ratio separately
- **WebSocket lifecycle**: Managed by TypeScript WebSocketManager class
- **TypeScript compilation**: esbuild produces single IIFE bundle with toolbar configuration
- **Download functionality**: HTTP-based endpoint using connection-scoped cache
- **File formats**: All matplotlib formats (PNG, PDF, SVG, EPS, PS, JPG, TIFF)
- **Build system**: Python setuptools orchestrates TypeScript compilation automatically
- **Development mode**: `pixi run dev` for hot reload (TypeScript watch + Python server)

## Known Limitations & Future Work

### Potential Improvements
- Rate limiting for WebSocket endpoints
- CORS configuration options
- Prometheus metrics for monitoring
- Configurable timeouts for plot generation
- Static plot metadata caching
- Session management with per-user state persistence

### Already Implemented
- Health check endpoints (`/health` public, `/health/details` authenticated)
- Async plot generation with a bounded thread pool

## Dependencies

### Runtime
- **Python**: >= 3.12
- **Python packages**: matplotlib >= 3.10.0, FastAPI >= 0.115.0, pydantic >= 2.0, numpy, Pillow, uvicorn
- **Node.js**: >= 20 (build-time requirement, not runtime)

### Development
- **Python tooling**: ruff (linting/formatting), mypy (type checking), pytest (testing)
- **JavaScript tooling**: TypeScript 5.3+, esbuild 0.20+
- **Environment**: pixi (manages both Python and Node.js dependencies)

## Common Patterns & Idioms

### PlotConfig Structure
```python
# Static plot (no updates)
PlotConfig(
    description="Static visualization",
    init=InitConfig(generator_func, ParamsModel)
)

# Interactive plot with updates
PlotConfig(
    description="Interactive visualization",
    init=InitConfig(generator_func, InitParamsModel),
    update=UpdateConfig(update_func, UpdateParamsModel)
)
```

### State Management Pattern
```python
# Generator returns state (can be any type - dict, dataclass, etc.)
def create_plot(fig: Figure, params: InitParams) -> dict:
    line, = ax.plot(x, y)
    return {"line": line, "data": x}

# Update receives and returns state
def update_plot(state: dict, params: UpdateParams) -> dict:
    state["line"].set_ydata(new_data)
    return state
```

### Parameter Validation with Constraints
```python
class PlotParams(BaseModel):
    value: float = Field(
        default=1.0,
        ge=0.1,      # minimum (greater or equal)
        le=10.0,     # maximum (less or equal)
        description="User-friendly description"
    )
```
Constraints automatically appear in HTML form and JSON schema.

### Callback Parameters
Some methods have unused parameters required by matplotlib's callback API:
```python
def method(self, event: Any, websocket: WebSocket) -> None:  # noqa: ARG002
    # event might be unused but required by API signature
```
Use `# noqa: ARG002` to suppress linter warnings.

### Logging Best Practices
```python
logger.info(f"Connection established: {plot_name}")
logger.warning(f"Invalid input: {error}")
logger.error(f"Failed to render: {e}", exc_info=True)
logger.debug(f"Event received: {event_type}")
```

## Testing & Development

### Development Workflow
```bash
# Start development server with hot reload
pixi run dev
# TypeScript watch mode runs in parallel with Python server
# Visit http://localhost:8000

# Manual TypeScript build
pixi run js-build

# Install JavaScript dependencies
pixi run js-install
```

### Run Tests
```bash
pixi run test
```

### Linting & Type Checking
```bash
# Run every check (format, lint, typecheck, js checks, tests, js-build)
pixi run check

# Individual Python checks
pixi run lint
pixi run typecheck

# Individual JS checks
pixi run js-lint
pixi run js-typecheck
```

See `AGENTS.md` for the full task table.

### Building for Distribution
```bash
# Build Python package (automatically builds TypeScript first)
pixi run build

# Clean build artifacts
rm -rf dist/ build/ *.egg-info mpl_fastapi/static/js/dist/
```

**Maintainer Notes**:
- Fully typed codebase: Python (mypy strict) and TypeScript (strict mode)
- All unused callback arguments properly marked with `# noqa: ARG002`
- No deprecated matplotlib APIs used
- Clean separation: backend, routing, and client layers
- State management enables efficient updates without reconnection
- WebSocketManager pattern eliminates race conditions
- Custom build backend integrates TypeScript into Python packaging
- Single unified component supports all use cases
- Production-ready: minified bundles, source maps, automated builds
- Development-friendly: hot reload, fast compilation, comprehensive types

---

**Last Updated**: February 2, 2026
**Python Version**: 3.12+
**Node.js Version**: 20+ (build-time only)
**Key Contributors**: See AUTHORS.rst

