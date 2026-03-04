# mpl-fastapi React Example

A minimal React application demonstrating how to use `mpl-fastapi` as an npm package,
with everything served through a single FastAPI server.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 FastAPI Server                   │
│                                                  │
│  /react-app/*     → React build (static files)  │
│  /mpl-static/*    → Matplotlib CSS/images       │
│  /plots/*         → Plot API + WebSocket        │
└─────────────────────────────────────────────────┘
```

Single server, no CORS configuration needed!

## Setup

### 1. Build the mpl-fastapi npm package

From the repository root:

```bash
npm run build:npm
```

### 2. Install React example dependencies

```bash
cd demos/react-example
npm install
```

### 3. Build the React app

```bash
npm run build
```

This outputs to `demos/react-example/dist/`.

### 4. Run the FastAPI server

From the repository root:

```bash
uvicorn demos.sine_wave:app --reload
```

### 5. Open in browser

Visit: http://localhost:8000/react-app/

## Development Workflow

For development with auto-rebuild:

```bash
# Terminal 1: Watch and rebuild React app
cd demos/react-example
npm run dev

# Terminal 2: Run FastAPI with reload
uvicorn demos.sine_wave:app --reload
```

## How It Works

1. **React imports mpl-fastapi**: The `package.json` uses `"mpl-fastapi": "file:../.."` 
   to reference the local package. Vite bundles it into the React app.

2. **MatplotlibPlot component**: A React wrapper around `MatplotlibEmbeddable` that:
   - Creates the plot on mount
   - Updates parameters via props
   - Cleans up on unmount

3. **Single server**: FastAPI serves:
   - The React build as static files at `/react-app/`
   - The matplotlib static files at `/mpl-static/`
   - The plot API and WebSocket at `/plots/`

4. **No proxy needed**: Since everything is same-origin, no CORS or proxy configuration required.

## Files

- `src/App.tsx` - Main React component with controls
- `src/MatplotlibPlot.tsx` - Reusable React wrapper for MatplotlibEmbeddable
- `src/main.tsx` - React entry point
- `vite.config.ts` - Vite build configuration
- `package.json` - Dependencies including local mpl-fastapi

## Using in Your Own Project

To use this pattern in your own project:

1. Install the package: `npm install mpl-fastapi` (once published) or use `file:` reference
2. Copy `MatplotlibPlot.tsx` to your project
3. Mount the React build in FastAPI as static files
4. Ensure matplotlib CSS is available (either via FastAPI static mount or bundled)
