# WebSocket Protocol Documentation

This document describes the matplotlib-fastapi WebSocket protocol for interactive plotting.

## Table of Contents

1. [Initial Connection Flow](#initial-connection-flow)
2. [Client-Initiated Messages](#client-initiated-messages)
3. [Server-Initiated Messages](#server-initiated-messages)
4. [Idle Draw Cycle](#idle-draw-cycle)

---

## Initial Connection Flow

The initial connection establishes the WebSocket, validates the protocol, and sends configuration to the client.

```mermaid
sequenceDiagram
    participant Client as Browser (TypeScript)
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas
    participant Toolbar as NavigationToolbar2FastAPI
    participant Executor as ThreadPoolExecutor

    Note over Client,Router: Connection Establishment
    Client->>Router: WebSocket connect to /ws/{plot_name}?{params}
    Router->>Router: Validate plot exists
    Router->>Router: Accept WebSocket connection
    
    Note over Router: Server sends protocol version first
    Router->>Client: {"type": "protocol_version", "version": 0}
    Router->>Router: Validate query parameters
    Router->>Executor: Run plot init function(fig, params)
    Executor-->>Router: Return state dict
    Router->>Router: Create FastAPICanvas(fig)
    Router->>Router: Create FastAPIManger(canvas)
    
    Note over Router: Wait for client protocol version (REQUIRED first message)
    Client->>Router: {"type": "protocol_version", "version": 0}
    Router->>Router: Validate client version
    
    Note over Router: After validation, send all configuration messages
    Router->>Client: {"type": "image_mode", "mode": "full"}
    Router->>Client: {"type": "connection_id", "id": "<uuid>"}
    Router->>Client: {"type": "toolbar_config", "items": [...]}
    Router->>Client: {"type": "save_formats", "formats": [...]}
    Router->>Client: {"type": "default_save_format", "format": "png"}
    Router->>Client: {"type": "history_buttons", "Back": false, "Forward": false}
    
    Note over Router: Enter event loop for normal communication
    Router->>Router: await websocket.receive_json()
    
    Note over Client: Client sends remaining initialization messages
    Client->>Router: {"type": "supports_binary", "value": true}
    Router->>Router: Set manager.supports_binary flag (no response)
    Router->>Router: Drain queue (empty)
    Router->>Router: await websocket.receive_json()
    
    Client->>Router: {"type": "send_image_mode"}
    Router->>Backend: handle_send_image_mode(ev, websocket)
    Backend->>Client: {"type": "image_mode", "mode": "full"}
    Router->>Router: Drain queue (empty)
    Router->>Router: await websocket.receive_json()
    
    Client->>Router: {"type": "set_device_pixel_ratio", "device_pixel_ratio": ratio}
    Router->>Backend: handle_set_device_pixel_ratio(ev, websocket)
    alt DPI changed
        Backend->>Client: {"type": "draw"}
    end
    Router->>Router: Drain queue (empty)
    Router->>Router: await websocket.receive_json()
    
    Client->>Router: {"type": "refresh"}
    Router->>Backend: handle_refresh(ev, websocket)
    Backend->>Client: {"type": "figure_label", "label": ""}
    Backend->>Client: {"type": "draw"}
    Router->>Router: Drain queue (empty)
    Router->>Router: await websocket.receive_json()
    
    Client->>Router: {"type": "draw"}
    Router->>Executor: _sync_draw_figure(canvas)
    Executor->>Backend: canvas.draw()
    Executor-->>Router: Return PNG bytes
    Router->>Client: {"type": "image_mode", "mode": "full|diff"}
    Router->>Client: <binary PNG data>
    Router->>Router: Drain queue (empty)
    Router->>Router: await websocket.receive_json()
    
    Note over Client,Router: Connection ready for interaction
```

### Key Points

1. **Protocol handshake is required**: 
   - Server sends `protocol_version` (version 0) immediately after accepting connection
   - Client MUST send `protocol_version` as its first message
   - Server validates version compatibility before sending any other messages
   - If incompatible, server sends error and closes connection (code 1008)

2. **Server sends 6 configuration messages** after protocol validation:
   - `image_mode` (full or diff)
   - `connection_id` (UUID for downloads)
   - `toolbar_config` (button definitions)
   - `save_formats` (available save formats)
   - `default_save_format` (default format)
   - `history_buttons` (initial toolbar state)

3. **Initialization is deterministic**: The toolbar's `set_history_buttons()` is deferred during `__init__` to prevent race conditions. After protocol validation, the server sends all configuration including history_buttons.

4. **Client sends remaining initialization messages** after handshake:
   - `supports_binary` - declares binary WebSocket support
   - `send_image_mode` - requests current image mode
   - `set_device_pixel_ratio` - sets device pixel ratio (if not 1.0)
   - `refresh` - requests initial draw
   - `draw` - triggers first image render

5. **Queue draining**: After handling each client message (except `draw` which handles responses inline), the router calls `canvas.drain_queue()` which sends all queued messages. Toolbar actions (pan, zoom, navigate) will queue messages that are sent at this point.

---

## Client-Initiated Messages

### Mouse Events

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas
    participant MPL as Matplotlib (button_press_event)
    participant Toolbar as Toolbar

    Client->>Router: {"type": "button_press", "x": x, "y": y, "button": n}
    Router->>Backend: handle_button_press(ev, websocket)
    Backend->>Backend: Convert coordinates (flip y-axis)
    Backend->>MPL: button_press_event(x, y, button, guiEvent=ev)
    Note over MPL,Toolbar: Matplotlib event handlers may trigger toolbar actions
    alt Toolbar activated
        Toolbar->>Backend: queue_event("navigate_mode", mode="PAN")
        Toolbar->>Backend: queue_event("message", message="Pan/Zoom mode")
    end
    Backend-->>Router: (handler returns)
    Router->>Router: Drain queue
    Router->>Client: {"type": "navigate_mode", "mode": "PAN"}
    Router->>Client: {"type": "message", "message": "Pan/Zoom mode"}
    Router->>Router: await next message
```

**Mouse event types:**
- `button_press` - Mouse button pressed
- `button_release` - Mouse button released
- `motion_notify` - Mouse moved
- `dblclick` - Double-click
- `scroll` - Mouse wheel scrolled
- `figure_enter` - Mouse entered figure
- `figure_leave` - Mouse left figure

**Response:** Queued messages (if any) from event handlers + toolbar

---

### Keyboard Events

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas
    participant MPL as Matplotlib

    Client->>Router: {"type": "key_press", "key": "a"}
    Router->>Backend: handle_key_press(ev, websocket)
    Backend->>MPL: key_press_event(key, guiEvent=ev)
    Backend-->>Router: (handler returns)
    Router->>Router: Drain queue (may have messages)
    Router->>Client: <queued messages if any>
    Router->>Router: await next message
```

**Keyboard event types:**
- `key_press` - Key pressed
- `key_release` - Key released

**Response:** Queued messages (if any) from event handlers

---

### Resize

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas
    participant Figure as Matplotlib Figure

    Client->>Router: {"type": "resize", "width": 800, "height": 600}
    Router->>Backend: handle_resize(ev, websocket)
    Backend->>Backend: Adjust for device_pixel_ratio
    Backend->>Figure: set_size_inches(w/dpi, h/dpi)
    Backend->>Client: {"type": "resize", "size": [w, h], "forward": true}
    Backend-->>Router: (handler returns)
    Router->>Router: Drain queue (empty)
    Router->>Router: await next message
    
    Note over Client: Client receives resize response
    Client->>Client: Update canvas dimensions
    alt forward is true
        Client->>Router: {"type": "refresh"}
        Note over Router,Backend: Triggers refresh flow (see below)
    end
```

**Request:**
```json
{"type": "resize", "width": 800, "height": 600}
```

**Response:** Direct send from handler
```json
{"type": "resize", "size": [800, 600], "forward": true}
```

---

### Set Device Pixel Ratio

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas

    Client->>Router: {"type": "set_device_pixel_ratio", "device_pixel_ratio": 2.0}
    Router->>Backend: handle_set_device_pixel_ratio(ev, websocket)
    Backend->>Backend: Update internal ratio
    alt Ratio changed
        Backend->>Backend: Mark _force_full = True
        Backend->>Client: {"type": "draw"}
    end
    Backend-->>Router: (handler returns)
    Router->>Router: Drain queue (empty)
    Router->>Router: await next message
    
    Note over Client: If draw message received
    Client->>Router: {"type": "draw"}
    Note over Router,Backend: Triggers draw flow (see Idle Draw Cycle)
```

**Request:**
```json
{"type": "set_device_pixel_ratio", "device_pixel_ratio": 2.0}
```

**Response:** Direct send (only if ratio changed)
```json
{"type": "draw"}
```

---

### Refresh

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas

    Client->>Router: {"type": "refresh"}
    Router->>Backend: handle_refresh(ev, websocket)
    Backend->>Client: {"type": "figure_label", "label": ""}
    Backend->>Backend: Set _force_full = True
    Backend->>Client: {"type": "draw"}
    Backend-->>Router: (handler returns)
    Router->>Router: Drain queue (empty)
    Router->>Router: await next message
    
    Note over Client: Client receives draw command
    Client->>Router: {"type": "draw"}
    Note over Router,Backend: Triggers draw flow (see Idle Draw Cycle)
```

**Request:**
```json
{"type": "refresh"}
```

**Response:** Direct sends from handler
```json
{"type": "figure_label", "label": "Figure Title"}
{"type": "draw"}
```

---

### Toolbar Button

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas
    participant Toolbar as NavigationToolbar2FastAPI

    Client->>Router: {"type": "toolbar_button", "name": "pan"}
    Router->>Backend: handle_toolbar_button(ev, websocket)
    Backend->>Toolbar: Call method by name (e.g., pan())
    Toolbar->>Backend: queue_event("navigate_mode", mode="PAN")
    Toolbar->>Backend: queue_event("message", message="Pan mode")
    Toolbar->>Backend: queue_event("history_buttons", Back=true, Forward=false)
    alt Method triggered redraw
        Backend->>Backend: draw_idle() -> queue_event("draw")
    end
    Backend-->>Router: (handler returns)
    Router->>Router: Drain queue
    Router->>Client: {"type": "navigate_mode", "mode": "PAN"}
    Router->>Client: {"type": "message", "message": "Pan mode"}
    Router->>Client: {"type": "history_buttons", "Back": true, "Forward": false}
    opt If draw queued
        Router->>Client: {"type": "draw"}
    end
    Router->>Router: await next message
```

**Common toolbar buttons:**
- `home` - Reset to original view
- `back` - Navigate back in view history
- `forward` - Navigate forward in view history
- `pan` - Activate pan/zoom mode
- `zoom` - Activate zoom mode

**Response:** Queued messages from toolbar (varies by button)

---

### Update Parameters

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas
    participant Executor as ThreadPoolExecutor
    participant UpdateFn as User Update Function

    Client->>Router: {"type": "update_params", "params": {"value": 2.5}}
    Router->>Router: Validate params against update schema
    alt Validation fails
        Router->>Client: {"type": "error", "message": "Invalid params"}
        Router->>Router: await next message
    else Validation succeeds
        Router->>Executor: Run update function(state, params)
        Executor->>UpdateFn: update_fn(fig, state, params)
        UpdateFn->>UpdateFn: Modify figure/state
        Executor-->>Router: Return updated state
        Router->>Backend: draw_idle()
        Backend->>Backend: queue_event("draw")
        Router->>Router: Drain queue
        Router->>Client: {"type": "draw"}
        Router->>Router: await next message
        
        Note over Client: Client receives draw command
        Client->>Router: {"type": "draw"}
        Note over Router,Backend: Triggers draw flow (see Idle Draw Cycle)
    end
```

**Request:**
```json
{"type": "update_params", "params": {"value": 2.5, "color": "red"}}
```

**Response:** Queued draw message
```json
{"type": "draw"}
```

---

### Save Figure

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Executor as ThreadPoolExecutor
    participant Figure as Matplotlib Figure

    Client->>Router: {"type": "save_figure", "format": "png", "dpi": 100}
    Router->>Router: Validate format
    Router->>Executor: _sync_save_figure(fig, format, dpi, transparent)
    Executor->>Figure: fig.savefig(buf, format=format, dpi=dpi)
    Executor-->>Router: Return bytes
    Router->>Router: Generate file_id, store file
    Router->>Client: {"type": "save_complete", "file_id": "...", "download_url": "..."}
    Router->>Router: Drain queue (empty)
    Router->>Router: await next message
```

**Request:**
```json
{
  "type": "save_figure",
  "format": "png",
  "dpi": 100,
  "transparent": false
}
```

**Response:** Direct send
```json
{
  "type": "save_complete",
  "file_id": "abc123",
  "download_url": "/plots/download/abc123",
  "filename": "figure.png",
  "format": "png"
}
```

---

## Server-Initiated Messages

The server can send messages at any time, typically in response to canvas events or toolbar actions.

### Message Types

| Type | Trigger | Content |
|------|---------|---------|
| `draw` | Canvas needs redraw | `{"type": "draw"}` |
| `image_mode` | Before sending image | `{"type": "image_mode", "mode": "full"\|"diff"}` |
| `figure_label` | Figure title changed | `{"type": "figure_label", "label": "Title"}` |
| `message` | Toolbar displays message | `{"type": "message", "message": "Text"}` |
| `navigate_mode` | Pan/zoom mode changed | `{"type": "navigate_mode", "mode": "PAN"\|"ZOOM"\|"NONE"}` |
| `history_buttons` | Nav stack changed | `{"type": "history_buttons", "Back": bool, "Forward": bool}` |
| `rubberband` | Selection in progress | `{"type": "rubberband", "x0": n, "y0": n, "x1": n, "y1": n}` |
| `save` | Toolbar save button | `{"type": "save"}` (triggers client save flow) |

---

## Idle Draw Cycle

The idle draw cycle is how matplotlib figures are rendered and sent to the browser.

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Executor as ThreadPoolExecutor
    participant Backend as FastAPICanvas
    participant Renderer as RendererAgg

    Note over Client,Backend: Something triggered canvas.draw_idle()
    Backend->>Backend: queue_event("draw")
    Note over Backend: Message queued, not sent yet
    
    Note over Router: After current event handler completes
    Router->>Router: Drain queue
    Router->>Client: {"type": "draw"}
    Router->>Router: await next message
    
    Note over Client: Client receives draw command
    Client->>Router: {"type": "draw"}
    
    Note over Router: Handle draw message
    Router->>Executor: _sync_draw_figure(canvas)
    Note over Executor: Run in background thread
    Executor->>Backend: get_renderer()
    Backend->>Backend: Check if new renderer needed
    alt Need new renderer
        Backend->>Renderer: Create RendererAgg(w, h, dpi)
        Backend->>Backend: Save last buffer for diff
    end
    Backend-->>Executor: Return renderer
    
    Executor->>Backend: canvas.draw()
    Backend->>Backend: Render all figure artists
    
    Executor->>Backend: Determine image mode (full or diff)
    alt First draw or _force_full
        Backend->>Backend: mode = "full"
        Backend->>Backend: Clear _force_full flag
    else Subsequent draw
        Backend->>Backend: Compare buffers
        alt > 50% pixels changed
            Backend->>Backend: mode = "full"
        else < 50% pixels changed
            Backend->>Backend: mode = "diff"
            Backend->>Backend: Create diff image (changed pixels only)
        end
    end
    
    Executor->>Executor: Encode PNG (full or diff)
    Executor-->>Router: Return PNG bytes
    
    Router->>Client: {"type": "image_mode", "mode": "full"}
    Router->>Client: <binary PNG data>
    Router->>Router: Drain queue (may have new messages)
    opt Queue not empty
        Router->>Client: <queued messages>
    end
    Router->>Router: await next message
    
    Note over Client: Client receives image
    Client->>Client: imageObj.src = blob URL
    Note over Client: imageObj.onload fires
    Client->>Client: context.drawImage(imageObj, 0, 0)
    Note over Client: Figure updated on screen
```

### Key Points

1. **Asynchronous queueing**: `draw_idle()` doesn't immediately send a message, it queues one. The queue is drained after the current event handler completes.

2. **Thread pool execution**: The actual drawing happens in a background thread to avoid blocking the event loop.

3. **Differential rendering**: After the first draw, the backend compares the current buffer with the previous one. If < 50% of pixels changed, it sends only the diff.

4. **Full vs Diff mode**:
   - **Full**: Entire PNG image
   - **Diff**: PNG with transparent background, only changed pixels visible

5. **Binary WebSocket**: Images are sent as binary WebSocket frames, not JSON.

6. **Client rendering**: The browser uses an `<img>` element to decode the PNG, then draws it to a `<canvas>` using `drawImage()`.

---

## Protocol Version Negotiation

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router

    Router->>Client: {"type": "protocol_version", "version": 0}
    Client->>Router: {"type": "protocol_version", "version": N}
    
    alt N == 0
        Note over Router: Version compatible, send configuration
        Router->>Client: 6 configuration messages
    else N == null or missing
        Router->>Client: {"type": "error", "message": "Protocol version is required"}
        Router->>Router: Close WebSocket (code 1008)
    else N != 0
        Router->>Client: {"type": "error", "message": "Incompatible protocol version"}
        Router->>Router: Close WebSocket (code 1008)
    end
```

**Protocol version is REQUIRED**: Both client and server must send the version field. The client's protocol_version message MUST be the first client message after receiving the server's protocol_version.

Currently, only protocol version `0` is supported. Future versions may introduce breaking changes to the message format or protocol flow.

---

## Error Handling

```mermaid
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router

    Client->>Router: Invalid message or parameters
    Router->>Router: Validation fails
    Router->>Client: {"type": "error", "message": "Error details"}
    
    alt Fatal error
        Router->>Router: Close WebSocket with error code
    else Recoverable error
        Router->>Router: Continue event loop
    end
```

**Error message format:**
```json
{"type": "error", "message": "Error description"}
```

Common error scenarios:
- Invalid protocol version
- Invalid parameters for plot initialization
- Invalid parameters for update
- Unknown save format
- Validation errors
- Protocol version missing or incompatible
- Wrong message type as first client message

---

## Message Queue Behavior

**Important:** The message queue is drained after MOST client messages are processed, with one exception:

1. **`draw`**: Sends complete response inline (image_mode + binary PNG), then continues to next message

**Protocol handshake is outside the event loop:** The protocol_version exchange happens before entering the main event loop, so it doesn't use the queue mechanism.

```python
# In router.py:

# Protocol handshake (before event loop):
# 1. Send server protocol_version
# 2. Wait for client protocol_version
# 3. Validate and send all configuration messages

# Event loop:
while True:
    data = await websocket.receive_json()
    
    # draw handler:
    if data["type"] == "draw":
        # ... render image ...
        await websocket.send_json({"type": "image_mode", ...})
        await websocket.send_bytes(image_data)
        continue  # Skip drain_queue

    # All other handlers:
    else:
        await handler(data, websocket)
        await canvas.drain_queue(websocket)  # Always called for other messages
```

This means:
1. Protocol handshake completes before entering event loop
2. Toolbar can queue messages at any time during event handlers
3. Those messages will be sent after the next client message is processed (except draw)
4. Tests must account for variable message sequences
4. Client should be prepared to receive queued messages after most requests
4. Client should be prepared to receive queued messages after most requests
