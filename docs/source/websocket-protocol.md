# WebSocket Protocol Documentation (v0)

This document describes the matplotlib-fastapi WebSocket protocol version 0 for interactive plotting.

## Table of Contents

1. [Protocol Overview](#protocol-overview)
2. [Initial Connection Flow](#initial-connection-flow)
3. [Binary Image Format](#binary-image-format)
4. [Client-Initiated Messages](#client-initiated-messages)
5. [Server-Initiated Messages](#server-initiated-messages)
6. [Render Cycle](#render-cycle)

---

## Protocol Overview

### Versioning

The WebSocket endpoint includes the protocol version in the URL path:
- **v0**: `/ws/v0/{plot_name}` (current)
- Legacy endpoint `/ws/{plot_name}` redirects to v0

### Key Design Principles

1. **Client initiates**: Client sends `init` message first
2. **Consolidated config**: Server responds with single `config` message
3. **Self-describing binary**: Images include 8-byte header with metadata
4. **Sequence tracking**: Image sequence numbers enable diff validation
5. **Stateful URIs**: Full figure state (init + update params) is encodable in the URL

### Query String Parameters

The WebSocket URL accepts two categories of query parameters:

- **Init parameters** (flat): `?frequency=2.0&amplitude=1.5` — validated against `InitConfig.params_model`
- **Update parameters** (prefixed): `?_update.phase=1.57` — validated against `UpdateConfig.params_model`

The `_update.` prefix is a reserved namespace. When present, the server:
1. Calls the init function with the flat params
2. Calls the update function with the `_update.*` params (prefix stripped)
3. Returns the fully-applied figure state

This enables **shareable URIs** that capture the full figure state:
```
/ws/v0/interactive_sine?frequency=2.0&amplitude=1.0&_update.phase=1.57
```

---

## Initial Connection Flow

The v0 protocol consolidates the handshake into just 2 messages:

```{mermaid}
sequenceDiagram
    participant Client as Browser (TypeScript)
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas
    participant Executor as ThreadPoolExecutor

    Note over Client,Router: Connection Establishment
    Client->>Router: WebSocket connect to /ws/v0/{plot_name}?{init_params}&_update.{key}={value}
    Router->>Router: Validate plot exists
    Router->>Router: Accept WebSocket connection

    Note over Client,Router: Client sends init (REQUIRED first message)
    Client->>Router: {"type": "init", "protocol_version": 0, "device_pixel_ratio": 2.0, "supports_binary": true}
    Router->>Router: Validate protocol version
    Router->>Router: Split query params (init vs _update.*)
    Router->>Executor: Run plot init function(fig, init_params)
    Executor-->>Router: Return state dict
    Router->>Router: If _update.* params present and update configured:
    Router->>Executor: Run update function(state, update_params)
    Executor-->>Router: Return updated state
    Router->>Router: Create FastAPICanvas(fig)
    Router->>Router: Create FastAPIManger(canvas)
    Router->>Router: Apply device_pixel_ratio

    Note over Router: Server sends consolidated config message
    Router->>Client: {"type": "config", "protocol_version": 0, "connection_id": "<uuid>", "figure": {...}, "toolbar": {...}, "save": {...}, "image": {...}, "update_schema": {...}}

    Note over Client: Client receives config and completes setup
    Client->>Client: Set canvas_div size from figure.size
    Client->>Client: Initialize toolbar from toolbar.items
    Client->>Client: Mark _initialized = true

    Note over Client,Router: Client requests first render
    Client->>Router: {"type": "refresh"}
    Router->>Executor: _sync_draw_figure(canvas)
    Executor-->>Router: Return (image_bytes, is_diff=false)
    Router->>Client: <8-byte header + PNG data>

    Note over Client,Router: Connection ready for interaction
```

### Message Details

#### Client `init` Message (Required)

```json
{
  "type": "init",
  "protocol_version": 0,
  "device_pixel_ratio": 2.0,
  "supports_binary": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | Yes | - | Must be "init" |
| `protocol_version` | int | Yes | - | Must match server version (0) |
| `device_pixel_ratio` | float | No | 1.0 | Device pixel ratio for HiDPI displays |
| `supports_binary` | bool | No | true | Whether client supports binary WebSocket messages |

#### Server `config` Message

```json
{
  "type": "config",
  "protocol_version": 0,
  "connection_id": "550e8400-e29b-41d4-a716-446655440000",
  "figure": {
    "size": [640, 480],
    "dpi": 100,
    "label": "My Figure"
  },
  "toolbar": {
    "items": [["Home", "Reset view", "home", "home"], ...],
    "history": {"back": false, "forward": false}
  },
  "save": {
    "formats": ["png", "pdf", "svg"],
    "default_format": "png"
  },
  "image": {
    "format": "png"
  },
  "update_schema": null,
  "init_params": {"frequency": 2.0, "amplitude": 1.0, "phase": 0.0, "points": 200},
  "update_params": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `init_params` | object | Serialized init parameters (always present) |
| `update_params` | object\|null | Serialized update parameters applied on init, or null |

### Error Handling

If the client sends an invalid `init` message, the server responds with an error and closes the connection:

```json
{"type": "error", "message": "Incompatible protocol version. Server: 0, client: 999"}
```

WebSocket close codes:
- `1008`: Policy violation (invalid plot, params, or protocol mismatch)
- `1011`: Internal server error (plot generation failed)

---

## Binary Image Format

All rendered images are sent as binary WebSocket messages with an 8-byte header.

### Header Structure

```
Offset  Size  Type     Field       Description
------  ----  ------   ----------  -----------
0       1     uint8    type_mode   0x00=FULL, 0x01=DIFF
1       1     uint8    format      0x01=PNG, 0x02=JPEG, 0x03=WebP
2       2     uint16   seq_num     Sequence number (1-65535, 0 reserved)
4       2     uint16   base_seq    Base sequence for diffs (0 for FULL)
6       2     uint16   flags       Reserved for future use
8       ...   bytes    image_data  Image file data
```

### Sequence Numbers

- Start at 1, increment for each image sent
- Wrap from 65535 to 1 (0 is reserved as sentinel)
- For FULL images: `base_seq` is 0
- For DIFF images: `base_seq` is the sequence number of the base image

### Client Handling

```typescript
// Parse binary message
const view = new DataView(buffer);
const typeMode = view.getUint8(0);  // 0=FULL, 1=DIFF
const format = view.getUint8(1);    // 1=PNG, 2=JPEG, 3=WebP
const seqNum = view.getUint16(2, false);  // big-endian
const baseSeq = view.getUint16(4, false);
const flags = view.getUint16(6, false);
const imageData = new Uint8Array(buffer, 8);

// Validate diff base if needed
if (typeMode === 0x01 && baseSeq !== lastSeqNum) {
  console.warn('Diff based on unknown frame, requesting full refresh');
  sendMessage({type: 'refresh'});
}
```

---

## Client-Initiated Messages

### Render Request

Request an image render.

```json
{"type": "render"}
```

**Response:** Binary image with header

---

### Refresh Request

Request a full (non-diff) render.

```json
{"type": "refresh"}
```

**Response:** Binary image with `type_mode=0x00` (FULL)

---

### Mouse Events

```json
{"type": "button_press", "x": 100, "y": 200, "button": 0, "buttons": 1}
{"type": "button_release", "x": 100, "y": 200, "button": 0, "buttons": 0}
{"type": "motion_notify", "x": 150, "y": 250, "button": -1, "buttons": 0}
{"type": "scroll", "x": 100, "y": 200, "button": 0, "buttons": 0, "step": 1}
{"type": "dblclick", "x": 100, "y": 200, "button": 0, "buttons": 1}
{"type": "figure_enter", "x": 0, "y": 0, "button": -1, "buttons": 0}
{"type": "figure_leave", "x": 0, "y": 0, "button": -1, "buttons": 0}
```

**Response:** Queued messages (if any) from event handlers

#### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `x`, `y` | yes | Cursor position in CSS pixels, origin at the **top-left** of the canvas. The server flips `y` to matplotlib's bottom-left origin. |
| `button` | **yes** | Browser [`MouseEvent.button`](https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button) value (`0`=left, `1`=middle, `2`=right, `3`=back, `4`=forward; `-1` when no button is associated, e.g. for motion). The server reads this on **every** mouse event and adds 1 to obtain the matplotlib button number, so it must always be present — including for `motion_notify`, `figure_enter`, and `figure_leave`. |
| `buttons` | no (default `0`) | Browser [`MouseEvent.buttons`](https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/buttons) bitmask of buttons currently held (`1`=left, `2`=right, `4`=middle, `8`=back, `16`=forward). Decoded into the matplotlib `buttons` set. |
| `step` | scroll only | Scroll amount for `scroll` events. |
| `modifiers` | no (default `[]`) | List of held modifier keys (`"ctrl"`, `"alt"`, `"shift"`, `"meta"`, …). |
| `guiEvent` | no | Opaque serialization of the raw DOM event, stored unmodified on the matplotlib event object. On `motion_notify` (which fires at a high rate) the reference client sends only the **user-input** fields (`altKey`, `ctrlKey`, `shiftKey`, `metaKey`, `button`, `buttons`) and drops the static positional/timing fields; on discrete events it sends the full serialization. Clients may send any subset, or omit it entirely. |

---

### Keyboard Events

```json
{"type": "key_press", "key": "ctrl+z"}
{"type": "key_release", "key": "a"}
```

**Response:** Queued messages (if any) from event handlers

---

### Resize

```json
{"type": "resize", "width": 800, "height": 600}
```

**Response:**
```json
{"type": "resize", "size": [800, 600], "forward": true}
```

---

### Toolbar Button

```json
{"type": "toolbar_button", "name": "pan"}
```

Common buttons: `home`, `back`, `forward`, `pan`, `zoom`

**Response:** Queued messages from toolbar (navigate_mode, history_buttons, etc.)

---

### Update Parameters

```json
{"type": "update_params", "params": {"frequency": 2.5}}
```

**Response:**
```json
{"type": "invalidate"}
```

---

### Save Figure

```json
{"type": "save_figure", "format": "png", "dpi": 150, "transparent": false}
```

**Response:**
```json
{
  "type": "save_complete",
  "file_id": "abc123",
  "download_url": "/plots/download/abc123",
  "filename": "figure.png",
  "format": "png"
}
```

Or on error:
```json
{"type": "save_error", "message": "Unsupported format"}
```

---

## Server-Initiated Messages

### Cursor Update

```json
{"type": "cursor", "cursor": "crosshair"}
```

### Status Message

```json
{"type": "message", "message": "x=1.5, y=2.3"}
```

### Navigate Mode

```json
{"type": "navigate_mode", "mode": "PAN"}
```

Mode values: `"PAN"`, `"ZOOM"`, `null`

### History Buttons

```json
{"type": "history_buttons", "Back": true, "Forward": false}
```

### Invalidate

Signals that the figure has changed and client should request a render.

```json
{"type": "invalidate"}
```

### Rubberband

Draw a selection rectangle (for zoom tool).

```json
{"type": "rubberband", "x0": 100, "y0": 200, "x1": 300, "y1": 400}
```

### Resize

Server-initiated resize (e.g., from figure.set_size_inches).

```json
{"type": "resize", "size": [800, 600], "forward": true}
```

### Error

```json
{"type": "error", "message": "Error description"}
```

---

## Render Cycle

### Idle Draw Pattern

```{mermaid}
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router
    participant Backend as FastAPICanvas

    Note over Backend: Some action triggers draw_idle()
    Backend->>Backend: queue_event("invalidate")
    Backend-->>Router: (handler returns)
    Router->>Router: Drain queue
    Router->>Client: {"type": "invalidate"}

    Note over Client: Client receives invalidate
    Client->>Router: {"type": "render"}
    Router->>Router: Execute _sync_draw_figure in thread pool
    Router->>Client: <8-byte header + PNG data>

    Note over Client: Client displays image
    Client->>Client: Parse header, create blob, update canvas
```

### Full Refresh Pattern

```{mermaid}
sequenceDiagram
    participant Client as Browser
    participant Router as FastAPI Router

    Client->>Router: {"type": "refresh"}
    Router->>Router: Set canvas._force_full = True
    Router->>Router: Execute _sync_draw_figure in thread pool
    Router->>Client: <8-byte header (type_mode=0x00 FULL) + PNG data>
```

---

## Message Count Comparison

| Action | Old Protocol | v0 Protocol |
|--------|--------------|-------------|
| Handshake | 14 messages | 5 messages |
| Render | 2 messages (image_mode + binary) | 1 message (binary with header) |
| Refresh | 3 messages (label + image_mode + binary) | 1 message (binary with header) |

The v0 protocol reduces initialization overhead by ~65% and per-render overhead by 50%.
