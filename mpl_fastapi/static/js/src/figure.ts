/**
 * Figure - Matplotlib figure renderer with WebSocket communication
 *
 * This class handles:
 * - Canvas rendering and image updates
 * - Mouse and keyboard event handling
 * - Toolbar integration
 * - WebSocket message protocol (v0)
 */

import type {
  ImageMode,
  ConfigMessage,
  ConnectionIdMessage,
  CursorMessage,
  ErrorMessage,
  FigureLabelMessage,
  ImageModeMessage,
  NavigateModeMessage,
  RubberbandMessage,
  SaveCompleteMessage,
  SaveErrorMessage,
  StatusMessage,
} from './types.js';
import {
  PROTOCOL_VERSION,
  ImageTypeMode,
  parseBinaryImage,
  getImageMimeType,
} from './types.js';
import { WebSocketManager } from './websocket-manager.js';

// Static path configuration
declare global {
  interface Window {
    MPL_STATIC_PATH?: string;
  }
}

/**
 * Utility: Get mouse position relative to element
 */
function findpos(e: MouseEvent): { x: number; y: number } {
  let targ = e.target as HTMLElement | null;
  if (targ && targ.nodeType === 3) {
    // Defeat Safari bug
    targ = targ.parentNode as HTMLElement;
  }

  if (!targ) {
    return { x: 0, y: 0 };
  }

  const boundingRect = targ.getBoundingClientRect();
  const x = e.clientX - boundingRect.left;
  const y = e.clientY - boundingRect.top;

  return { x, y };
}

/**
 * Utility: Extract non-object keys to avoid circular references
 */
function simpleKeys(original: Record<string, any>): Record<string, any> {
  return Object.keys(original).reduce((obj: Record<string, any>, key) => {
    if (typeof original[key] !== 'object') {
      obj[key] = original[key];
    }
    return obj;
  }, {});
}

/**
 * Main Figure class for rendering matplotlib plots
 */
export class Figure {
  // Public properties
  readonly id: string;
  connection_id: string | null = null;
  readonly ws_manager: WebSocketManager;
  ws: WebSocket | null = null; // Legacy property

  // Canvas and rendering
  canvas: HTMLCanvasElement | undefined;
  canvas_div: HTMLDivElement | undefined;
  context: CanvasRenderingContext2D | undefined;
  rubberband_canvas: HTMLCanvasElement | undefined;
  rubberband_context: CanvasRenderingContext2D | undefined;
  ratio: number = 1;
  image_mode: ImageMode = 'full';
  readonly imageObj: HTMLImageElement;
  waiting: boolean = false;

  // UI elements
  readonly root: HTMLDivElement;
  header: HTMLDivElement | undefined;
  message: HTMLSpanElement | undefined;
  format_dropdown: HTMLSelectElement | undefined;
  buttons: Record<string, HTMLButtonElement> = {};

  // Toolbar configuration (received via WebSocket)
  private toolbar_items: Array<[string, string, string, string]> = [];
  private save_formats: string[] = [];
  private default_save_format: string = 'png';
  private toolbar_ready: boolean = false;

  // State
  supports_binary: boolean = true;
  private _key: string | null = null;
  private resizeObserverInstance: ResizeObserver | null = null;
  private _resize_canvas?: (width: number, height: number, forward: boolean) => void;
  private _server_size: [number, number] | null = null;
  private _initialized: boolean = false;
  private _initial_size: [number, number] | null = null;

  // Echoed params from config message (for reconstructing state URLs)
  server_init_params: Record<string, unknown> = {};
  server_update_params: Record<string, unknown> | null = null;

  // Reconnection state
  private _reconnecting: boolean = false;
  private _reconnect_overlay: HTMLDivElement | null = null;
  private _reconnect_label: HTMLDivElement | null = null;

  // Download handler
  ondownload: (fig: Figure, format: string) => void;

  /**
   * Create and attach a new Figure to the DOM.
   *
   * @param figure_id - Unique identifier for this figure (used in WebSocket messages).
   * @param ws_manager - The ``WebSocketManager`` that drives this figure's connection.
   * @param parent_element - DOM element that the figure's root div will be appended to.
   */
  constructor(
    figure_id: string,
    ws_manager: WebSocketManager,
    parent_element: HTMLElement
  ) {
    this.id = figure_id;
    this.ws_manager = ws_manager;

    this.imageObj = new Image();

    this.root = document.createElement('div');
    this.root.setAttribute('style', 'display: inline-block');
    this._root_extra_style(this.root);

    parent_element.appendChild(this.root);

    this._init_header();
    this._init_canvas();
    // Defer toolbar init until we receive config via WebSocket

    // Register open handler with WebSocketManager
    this.ws_manager.onOpen(() => {
      // Update legacy ws property
      this.ws = this.ws_manager.rawSocket;
      this.supports_binary = this.ws?.binaryType !== undefined;

      if (!this.supports_binary) {
        const warnings = document.getElementById('mpl-warnings');
        if (warnings) {
          warnings.style.display = 'block';
          warnings.textContent =
            'This browser does not support binary websocket messages. ' +
            'Performance may be slow.';
        }
      }

      // Send consolidated init message (v0 protocol - REQUIRED as first message)
      this.send_message('init', {
        protocol_version: PROTOCOL_VERSION,
        device_pixel_ratio: this.ratio,
        supports_binary: this.supports_binary,
      });

      // Server will respond with consolidated 'config' message
      // Then we send 'refresh' to get the first image
    });

    // Register reconnection handlers
    this.ws_manager.onReconnecting((attempt, maxAttempts) => {
      this._reconnecting = true;
      this._showOverlay(`Reconnecting\u2026  (${attempt}/${maxAttempts})`);
    });

    this.ws_manager.onReconnected(() => {
      // Reset canvas state before the onOpen handler re-sends init.
      // The fresh server-side figure will send a new config message;
      // handle_config will resize the server figure to match the
      // current canvas_div and request a refresh.
      this._reconnecting = true;
      this.connection_id = null;
      this.image_mode = 'full';
      this.waiting = false;
      this._server_size = null;

      // Clear the canvas so the user sees a blank plot rather than a
      // stale image at the (possibly wrong) size.
      if (this.context && this.canvas) {
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      }
    });

    this.ws_manager.onReconnectFailed(() => {
      this._reconnecting = false;
      this._showOverlay('Disconnected');
    });

    // Register message handler with WebSocketManager
    this.ws_manager.onMessage(this._make_on_message_function());

    this.imageObj.onload = () => {
      if (!this.context || !this.canvas) return;

      if (this.image_mode === 'full') {
        // Full images could contain transparency, clear canvas to avoid ghosting
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      }
      this.context.drawImage(this.imageObj, 0, 0);
    };

    this.imageObj.addEventListener('unload', () => {
      this.ws_manager.close();
    });

    this.ondownload = this._default_download_handler.bind(this);
  }

  private _init_header(): void {
    const titlebar = document.createElement('div');
    titlebar.className =
      'ui-dialog-titlebar ui-widget-header ui-corner-all ui-helper-clearfix';
    const titletext = document.createElement('div');
    titletext.className = 'ui-dialog-title';
    titletext.setAttribute('style', 'width: 100%; text-align: center; padding: 3px;');
    titlebar.appendChild(titletext);
    this.root.appendChild(titlebar);
    this.header = titletext;
  }

  /**
   * Extension point for subclasses to apply additional inline styles to the
   * figure's canvas wrapper ``<div>``.
   *
   * @param _canvas_div - The canvas container element.
   */
  protected _canvas_extra_style(_canvas_div: HTMLDivElement): void {
    // Hook for subclasses
  }

  /**
   * Extension point for subclasses to apply additional inline styles to the
   * figure's root ``<div>``.
   *
   * @param _root_div - The root container element.
   */
  protected _root_extra_style(_root_div: HTMLDivElement): void {
    // Hook for subclasses
  }

  private _init_canvas(): void {
    const canvas_div = (this.canvas_div = document.createElement('div'));
    canvas_div.setAttribute(
      'style',
      'border: 1px solid #ddd;' +
        'box-sizing: content-box;' +
        'clear: both;' +
        'min-height: 1px;' +
        'min-width: 1px;' +
        'outline: 0;' +
        'overflow: hidden;' +
        'position: relative;' +
        'resize: both;'
    );
    canvas_div.setAttribute('tabindex', '0');

    const on_keyboard_event_closure = (name: string) => {
      return (event: KeyboardEvent) => {
        return this.key_event(event, name);
      };
    };

    canvas_div.addEventListener('keydown', on_keyboard_event_closure('key_press'));
    canvas_div.addEventListener('keyup', on_keyboard_event_closure('key_release'));

    this._canvas_extra_style(canvas_div);
    this.root.appendChild(canvas_div);

    const canvas = (this.canvas = document.createElement('canvas'));
    canvas.classList.add('mpl-canvas');
    canvas.setAttribute('style', 'box-sizing: content-box;');

    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Failed to get 2D context');
    }
    this.context = context;

    this.ratio = window.devicePixelRatio || 1;

    const rubberband_canvas = (this.rubberband_canvas =
      document.createElement('canvas'));
    rubberband_canvas.setAttribute(
      'style',
      'box-sizing: content-box; position: absolute; left: 0; top: 0; z-index: 1;'
    );

    // ResizeObserver is available in all modern browsers (baseline 2020).
    // The _JSXTOOLS_RESIZE_OBSERVER ponyfill path is kept for niche
    // environments but should rarely (if ever) be needed.
    const RO: typeof ResizeObserver | undefined =
      window.ResizeObserver ??
      (window as any)._JSXTOOLS_RESIZE_OBSERVER?.({}).ResizeObserver;

    if (!RO) {
      console.warn('ResizeObserver not available');
      return;
    }

    this.resizeObserverInstance = new RO((entries: ResizeObserverEntry[]) => {
      for (const entry of entries) {
        let width: number, height: number;

        // contentBoxSize is always a ReadonlyArray per the spec.
        // Fall back to contentRect for very old polyfills.
        const cbs = entry.contentBoxSize?.[0];
        if (cbs) {
          width = cbs.inlineSize;
          height = cbs.blockSize;
        } else {
          width = entry.contentRect.width;
          height = entry.contentRect.height;
        }

        // Keep canvas and rubberband canvas in sync.
        // devicePixelContentBoxSize gives physical pixels directly,
        // avoiding rounding errors on HiDPI displays.
        const dpcs = entry.devicePixelContentBoxSize?.[0];
        if (dpcs) {
          canvas.setAttribute('width', String(dpcs.inlineSize));
          canvas.setAttribute('height', String(dpcs.blockSize));
        } else {
          canvas.setAttribute('width', String(width * this.ratio));
          canvas.setAttribute('height', String(height * this.ratio));
        }
        canvas.setAttribute('style', `width: ${width}px; height: ${height}px;`);

        rubberband_canvas.setAttribute('width', String(width));
        rubberband_canvas.setAttribute('height', String(height));

        // Only send resize to server if:
        // 1. We've completed initialization (received figure_size and done first render)
        // 2. This is NOT the size server just told us (prevents feedback loops)
        // 3. WebSocket is ready
        const isServerSize =
          this._server_size &&
          Math.abs(this._server_size[0] - width) < 1 &&
          Math.abs(this._server_size[1] - height) < 1;

        if (
          this._initialized &&
          this.ws_manager.isConnected() &&
          width !== 0 &&
          height !== 0 &&
          !isServerSize
        ) {
          this.request_resize(width, height);
        }
      }
    });
    if (this.resizeObserverInstance) {
      this.resizeObserverInstance.observe(canvas_div);
    }

    const on_mouse_event_closure = (name: string) => {
      return (event: MouseEvent) => {
        return this.mouse_event(event, name);
      };
    };

    rubberband_canvas.addEventListener(
      'mousedown',
      on_mouse_event_closure('button_press')
    );
    rubberband_canvas.addEventListener(
      'mouseup',
      on_mouse_event_closure('button_release')
    );
    rubberband_canvas.addEventListener('dblclick', on_mouse_event_closure('dblclick'));
    rubberband_canvas.addEventListener(
      'mousemove',
      on_mouse_event_closure('motion_notify')
    );
    rubberband_canvas.addEventListener(
      'mouseenter',
      on_mouse_event_closure('figure_enter')
    );
    rubberband_canvas.addEventListener(
      'mouseleave',
      on_mouse_event_closure('figure_leave')
    );

    canvas_div.addEventListener('wheel', (event: WheelEvent) => {
      (event as any).step = event.deltaY < 0 ? 1 : -1;
      on_mouse_event_closure('scroll')(event as any);
    });

    canvas_div.appendChild(canvas);
    canvas_div.appendChild(rubberband_canvas);

    const rubberband_context = rubberband_canvas.getContext('2d');
    if (!rubberband_context) {
      throw new Error('Failed to get rubberband 2D context');
    }
    this.rubberband_context = rubberband_context;
    this.rubberband_context.strokeStyle = '#000000';

    this._resize_canvas = (width: number, height: number, forward: boolean) => {
      if (forward && canvas_div) {
        // Store the size the server told us to use
        this._server_size = [width, height];
        canvas_div.style.width = width + 'px';
        canvas_div.style.height = height + 'px';
      }
    };

    // Disable right mouse context menu
    rubberband_canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      return false;
    });

    // Set focus after a delay
    setTimeout(() => {
      if (canvas) canvas.focus();
      if (canvas_div) canvas_div.focus();
    }, 100);
  }

  private _init_toolbar(): void {
    const toolbar = document.createElement('div');
    toolbar.className = 'mpl-toolbar';
    this.root.appendChild(toolbar);

    const on_click_closure = (name: string) => {
      return () => {
        return this.toolbar_button_onclick(name);
      };
    };

    const on_mouseover_closure = (tooltip: string) => {
      return (event: MouseEvent) => {
        const target = event.currentTarget as HTMLButtonElement;
        if (!target.disabled) {
          return this.toolbar_button_onmouseover(tooltip);
        }
      };
    };

    this.buttons = {};
    let buttonGroup = document.createElement('div');
    buttonGroup.className = 'mpl-button-group';

    // Use instance variable instead of window.mpl
    for (const [name, tooltip, image, method_name] of this.toolbar_items) {
      if (!name) {
        // Start a new button group
        if (buttonGroup.hasChildNodes()) {
          toolbar.appendChild(buttonGroup);
        }
        buttonGroup = document.createElement('div');
        buttonGroup.className = 'mpl-button-group';
        continue;
      }

      const button = (this.buttons[name] = document.createElement('button'));
      button.className = 'mpl-widget';
      button.setAttribute('role', 'button');
      button.setAttribute('aria-disabled', 'false');
      button.addEventListener('click', on_click_closure(method_name));
      button.addEventListener('mouseover', on_mouseover_closure(tooltip) as any);

      const icon_img = document.createElement('img');
      const static_path = window.MPL_STATIC_PATH || '/static';
      icon_img.src = `${static_path}/images/${image}.png`;
      icon_img.srcset = `${static_path}/images/${image}_large.png 2x`;
      icon_img.alt = tooltip;
      button.appendChild(icon_img);

      buttonGroup.appendChild(button);
    }

    if (buttonGroup.hasChildNodes()) {
      toolbar.appendChild(buttonGroup);
    }

    const fmt_picker = document.createElement('select');
    fmt_picker.className = 'mpl-widget';
    toolbar.appendChild(fmt_picker);
    this.format_dropdown = fmt_picker;

    // Use instance variables instead of window.mpl
    for (const fmt of this.save_formats) {
      const option = document.createElement('option');
      option.selected = fmt === this.default_save_format;
      option.innerHTML = fmt;
      fmt_picker.appendChild(option);
    }

    const status_bar = document.createElement('span');
    status_bar.className = 'mpl-message';
    toolbar.appendChild(status_bar);
    this.message = status_bar;
  }

  /**
   * Request that the server resize the figure canvas.
   *
   * Clears the locally-cached server size so that the ResizeObserver will
   * accept the server's response without treating it as a feedback loop.
   *
   * @param x_pixels - Desired canvas width in CSS pixels.
   * @param y_pixels - Desired canvas height in CSS pixels.
   */
  request_resize(x_pixels: number, y_pixels: number): void {
    this._server_size = null; // Clear so we accept the server's response
    this.send_message('resize', { width: x_pixels, height: y_pixels });
  }

  /**
   * Send a JSON message to the server over the WebSocket.
   *
   * Automatically injects ``type`` and ``figure_id`` fields.
   *
   * @param type - Message type string (e.g. ``'resize'``, ``'toolbar_button'``).
   * @param properties - Additional properties to include in the message.
   */
  send_message(type: string, properties: Record<string, any>): void {
    properties['type'] = type;
    properties['figure_id'] = this.id;
    this.ws_manager.send(JSON.stringify(properties));
  }

  /**
   * Request the server to render and send a new image frame.
   *
   * The ``waiting`` flag prevents duplicate in-flight render requests.
   */
  send_render_request(): void {
    if (!this.waiting) {
      this.waiting = true;
      this.ws_manager.send(JSON.stringify({ type: 'render', figure_id: this.id }));
    }
  }

  /**
   * Read an update form by ID and send its values to the server as an
   * ``update_params`` message.
   *
   * @param form_id - The ``id`` attribute of the ``<form>`` element to read.
   */
  send_update(form_id: string): void {
    const form = document.getElementById(form_id);
    if (!form) {
      console.error('Update form not found:', form_id);
      return;
    }

    const params: Record<string, unknown> = {};
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach((input) => {
      const inputEl = input as HTMLInputElement | HTMLSelectElement;
      let value: unknown = inputEl.value;

      if ((inputEl as HTMLInputElement).type === 'number') {
        value = parseFloat(inputEl.value);
      }
      params[inputEl.name] = value;
    });

    this.send_message('update_params', { params });

    // Track current update params locally
    this.server_update_params = params;
  }

  /**
   * Initiate a save/download by sending a ``save_figure`` WebSocket message.
   *
   * The format is read from the figure's format dropdown.
   *
   * @param fig - The Figure instance to save.
   * @param _msg - Unused (retained for handler signature compatibility).
   */
  handle_save(fig: Figure, _msg: unknown): void {
    if (!fig.format_dropdown) return;
    const selectedOption =
      fig.format_dropdown.options[fig.format_dropdown.selectedIndex];
    if (!selectedOption) return;
    const format = selectedOption.value;

    // Send save request via WebSocket instead of direct HTTP download
    fig.send_message('save_figure', {
      format: format,
      dpi: 100,
      transparent: false,
    });
  }

  /**
   * Handle a successful save response from the server.
   *
   * Triggers a browser download of the file at the URL returned by the server
   * and temporarily shows a confirmation in the status bar.
   *
   * @param fig - The Figure instance.
   * @param msg - Save-complete message containing ``download_url`` and ``filename``.
   */
  handle_save_complete(fig: Figure, msg: SaveCompleteMessage): void {
    const download_url = msg.download_url;
    const filename = msg.filename;

    // Trigger browser download (no DOM append needed in modern browsers)
    const link = document.createElement('a');
    link.href = download_url;
    link.download = filename;
    link.click();

    // Optional: Show success message
    if (fig.message) {
      fig.message.textContent = `Downloaded ${filename}`;
      // Clear message after 3 seconds
      setTimeout(() => {
        if (fig.message) {
          fig.message.textContent = '';
        }
      }, 3000);
    }
  }

  /**
   * Handle a save-error response from the server.
   *
   * Logs the error, displays it in the status bar for 5 seconds, and falls
   * back to an ``alert()`` if no status bar element exists.
   *
   * @param fig - The Figure instance.
   * @param msg - Save-error message containing the error ``message`` string.
   */
  handle_save_error(fig: Figure, msg: SaveErrorMessage): void {
    const error_message = msg.message;
    console.error('Save error:', error_message);

    // Show error to user
    if (fig.message) {
      fig.message.textContent = `Save failed: ${error_message}`;
      fig.message.style.color = 'red';
      // Clear message after 5 seconds
      setTimeout(() => {
        if (fig.message) {
          fig.message.textContent = '';
          fig.message.style.color = '';
        }
      }, 5000);
    } else {
      alert(`Failed to save figure: ${error_message}`);
    }
  }

  private _default_download_handler(fig: Figure, format: string): void {
    // This method is now deprecated but kept for backward compatibility
    // The new flow uses handle_save which sends save_figure message via WebSocket
    console.warn(
      '_default_download_handler is deprecated, using WebSocket save instead'
    );
    fig.send_message('save_figure', {
      format: format,
      dpi: 100,
      transparent: false,
    });
  }

  /**
   * Handle a resize notification from the server.
   *
   * Resizes the canvas to the dimensions specified by the server and requests
   * a new render frame.
   *
   * @param fig - The Figure instance.
   * @param msg - Message containing ``size`` ([width, height]) and ``forward`` flag.
   */
  handle_resize(fig: Figure, msg: any): void {
    const size = msg['size'];
    if (!fig.canvas || !fig._resize_canvas) return;

    if (size[0] !== fig.canvas.width || size[1] !== fig.canvas.height) {
      fig._resize_canvas(size[0], size[1], msg['forward']);
      fig.send_render_request();
    }
  }

  /**
   * Handle a rubberband selection rectangle from the server.
   *
   * Clears the rubberband canvas and redraws the selection outline.
   *
   * @param fig - The Figure instance.
   * @param msg - Rubberband coordinates in figure-space pixels.
   */
  handle_rubberband(fig: Figure, msg: RubberbandMessage): void {
    if (!fig.canvas || !fig.rubberband_context) return;

    let x0 = msg.x0 / fig.ratio;
    let y0 = (fig.canvas.height - msg.y0) / fig.ratio;
    let x1 = msg.x1 / fig.ratio;
    let y1 = (fig.canvas.height - msg.y1) / fig.ratio;

    x0 = Math.floor(x0) + 0.5;
    y0 = Math.floor(y0) + 0.5;
    x1 = Math.floor(x1) + 0.5;
    y1 = Math.floor(y1) + 0.5;

    const min_x = Math.min(x0, x1);
    const min_y = Math.min(y0, y1);
    const width = Math.abs(x1 - x0);
    const height = Math.abs(y1 - y0);

    fig.rubberband_context.clearRect(
      0,
      0,
      fig.canvas.width / fig.ratio,
      fig.canvas.height / fig.ratio
    );

    fig.rubberband_context.strokeRect(min_x, min_y, width, height);
  }

  /**
   * Update the figure title bar with a new label.
   *
   * @param fig - The Figure instance.
   * @param msg - Message containing the new ``label`` string.
   */
  handle_figure_label(fig: Figure, msg: FigureLabelMessage): void {
    if (fig.header) {
      fig.header.textContent = msg.label;
    }
  }

  /**
   * Update the CSS cursor style on the rubberband canvas.
   *
   * @param fig - The Figure instance.
   * @param msg - Message containing the CSS ``cursor`` string.
   */
  handle_cursor(fig: Figure, msg: CursorMessage): void {
    if (fig.rubberband_canvas) {
      fig.rubberband_canvas.style.cursor = msg.cursor;
    }
  }

  /**
   * Display a status/info message in the toolbar's message bar.
   *
   * @param fig - The Figure instance.
   * @param msg - Message containing the ``message`` text string.
   */
  handle_message(fig: Figure, msg: StatusMessage): void {
    if (fig.message) {
      fig.message.textContent = msg.message;
    }
  }

  /**
   * Handle consolidated config message (v0 protocol)
   * This replaces separate protocol_version, connection_id, toolbar_config,
   * save_formats, default_save_format, history_buttons, and figure_size messages.
   */
  handle_config(fig: Figure, msg: ConfigMessage): void {
    // Validate protocol version
    if (msg.protocol_version !== PROTOCOL_VERSION) {
      console.error(
        `Protocol version mismatch: client=${PROTOCOL_VERSION}, server=${msg.protocol_version}`
      );
      fig.ws_manager.close();
      throw new Error(
        `Incompatible protocol version. Client: ${PROTOCOL_VERSION}, server: ${msg.protocol_version}`
      );
    }
    console.log(`Protocol version validated: ${msg.protocol_version}`);

    // Store connection ID
    fig.connection_id = msg.connection_id;

    // Store echoed params (for reconstructing state URLs)
    fig.server_init_params = msg.init_params ?? {};
    fig.server_update_params = msg.update_params ?? null;

    // Store toolbar config (will init toolbar after)
    fig.toolbar_items = msg.toolbar.items;
    fig.save_formats = msg.save.formats;
    fig.default_save_format = msg.save.default_format;

    // Initialize toolbar now that we have all config
    if (!fig.toolbar_ready) {
      fig.toolbar_ready = true;
      fig._init_toolbar();
    }

    // Apply initial history button state
    if (fig.buttons['Back']) {
      fig.buttons['Back'].disabled = !msg.toolbar.history.back;
      fig.buttons['Back'].setAttribute(
        'aria-disabled',
        String(!msg.toolbar.history.back)
      );
    }
    if (fig.buttons['Forward']) {
      fig.buttons['Forward'].disabled = !msg.toolbar.history.forward;
      fig.buttons['Forward'].setAttribute(
        'aria-disabled',
        String(!msg.toolbar.history.forward)
      );
    }

    // Apply figure size
    const [width, height] = msg.figure.size;
    fig._initial_size = [width, height];
    fig._server_size = [width, height];

    // Set initial size on canvas
    if (fig.canvas_div) {
      fig.canvas_div.style.width = `${width}px`;
      fig.canvas_div.style.height = `${height}px`;
    }

    // Set figure label
    if (fig.header && msg.figure.label) {
      fig.header.textContent = msg.figure.label;
    }

    // Mark as initialized before first refresh to prevent ResizeObserver feedback
    fig._initialized = true;

    if (fig._reconnecting) {
      // On reconnect, the server created a default-sized figure.  Tell
      // it to match the *current* canvas_div size (which may differ
      // from the default) before requesting a refresh.  This mirrors
      // the Qt client's _on_reconnected logic.
      if (fig.canvas_div) {
        const curW = fig.canvas_div.clientWidth;
        const curH = fig.canvas_div.clientHeight;
        // Only send resize if the size actually differs from the config
        if (Math.abs(curW - width) > 1 || Math.abs(curH - height) > 1) {
          fig.request_resize(curW, curH);
        }
      }
    }

    // Request initial render now that canvas is properly sized
    fig.send_message('refresh', {});

    // If reconnecting, the overlay will be hidden when the first
    // binary image arrives (see _make_on_message_function).
  }

  /**
   * Handle error message from server
   */
  handle_error(fig: Figure, msg: ErrorMessage): void {
    console.error('Server error:', msg.message);
    if (fig.message) {
      fig.message.textContent = `Error: ${msg.message}`;
      fig.message.style.color = 'red';
    }
  }

  /**
   * Validate the server protocol version.
   *
   * @deprecated Replaced by the consolidated ``config`` message in v0 protocol.
   *   Kept for backward compatibility with older servers.
   * @param fig - The Figure instance.
   * @param msg - Message containing ``version`` number.
   * @throws {Error} If the server protocol version does not match the client.
   */
  handle_protocol_version(fig: Figure, msg: any): void {
    const server_version = msg['version'];
    // Protocol version is REQUIRED
    if (server_version == null) {
      console.error('Protocol version missing from server message');
      fig.ws_manager.close();
      throw new Error('Protocol version is required');
    }
    if (server_version !== PROTOCOL_VERSION) {
      console.error(
        `Protocol version mismatch: client expects ${PROTOCOL_VERSION}, server sent ${server_version}`
      );
      fig.ws_manager.close();
      throw new Error(
        `Incompatible protocol version. Client expects ${PROTOCOL_VERSION}, got ${server_version}`
      );
    }
    console.log(`Server protocol version validated: ${server_version}`);
  }

  /**
   * Handle an invalidation notification from the server.
   *
   * Triggers a new render request, allowing the client to pull the latest
   * frame after server-side state has changed.
   *
   * @param fig - The Figure instance.
   * @param _msg - Unused payload.
   */
  handle_invalidate(fig: Figure, _msg: unknown): void {
    fig.send_render_request();
  }

  /**
   * Update the local image mode (``'full'`` or ``'diff'``).
   *
   * @param fig - The Figure instance.
   * @param msg - Message containing the new ``mode`` string.
   */
  handle_image_mode(fig: Figure, msg: ImageModeMessage): void {
    fig.image_mode = msg.mode;
  }

  /**
   * Store the server-assigned connection ID.
   *
   * @deprecated Replaced by the ``connection_id`` field in the ``config`` message.
   * @param fig - The Figure instance.
   * @param msg - Message containing the ``id`` string.
   */
  handle_connection_id(fig: Figure, msg: ConnectionIdMessage): void {
    fig.connection_id = msg.id;
  }

  /**
   * Apply an initial figure size received from the server (legacy protocol).
   *
   * @deprecated Replaced by the ``figure.size`` field in the ``config`` message.
   * @param fig - The Figure instance.
   * @param msg - Message containing ``size`` ([width, height]) array.
   */
  // Legacy handler - config message now includes figure size
  handle_figure_size(fig: Figure, msg: any): void {
    // Store initial size to be applied before first render
    const size: [number, number] = msg['size'];
    fig._initial_size = size;
    fig._server_size = size;

    // Set initial size on canvas before requesting first render
    if (fig.canvas_div && fig._initial_size) {
      const [width, height] = fig._initial_size;
      fig.canvas_div.style.width = `${width}px`;
      fig.canvas_div.style.height = `${height}px`;
    }

    // Mark as initialized before first refresh to prevent ResizeObserver feedback
    fig._initialized = true;

    // Request initial render now that canvas is properly sized
    // Only do this for legacy protocol - v0 config handler does this
    if (!fig.toolbar_ready) {
      // Legacy path - need to wait for toolbar config
      fig.send_message('supports_binary', { value: fig.supports_binary });
      fig.send_message('send_image_mode', {});
      fig.send_message('refresh', {});
    }
  }

  /**
   * Store toolbar item configuration and initialise the toolbar when ready.
   *
   * @deprecated Replaced by the ``toolbar`` field in the ``config`` message.
   * @param fig - The Figure instance.
   * @param msg - Message containing ``items`` array.
   */
  handle_toolbar_config(fig: Figure, msg: any): void {
    fig.toolbar_items = msg['items'] as Array<[string, string, string, string]>;
    fig._check_toolbar_ready();
  }

  /**
   * Store supported save formats and initialise the toolbar when ready.
   *
   * @deprecated Replaced by the ``save.formats`` field in the ``config`` message.
   * @param fig - The Figure instance.
   * @param msg - Message containing ``formats`` array.
   */
  handle_save_formats(fig: Figure, msg: any): void {
    fig.save_formats = msg['formats'] as string[];
    fig._check_toolbar_ready();
  }

  /**
   * Store the default save format and initialise the toolbar when ready.
   *
   * @deprecated Replaced by the ``save.default_format`` field in the ``config`` message.
   * @param fig - The Figure instance.
   * @param msg - Message containing the ``format`` string.
   */
  handle_default_save_format(fig: Figure, msg: any): void {
    fig.default_save_format = msg['format'] as string;
    fig._check_toolbar_ready();
  }

  private _check_toolbar_ready(): void {
    // Initialize toolbar once we have all config
    if (
      !this.toolbar_ready &&
      this.toolbar_items.length > 0 &&
      this.save_formats.length > 0
    ) {
      this.toolbar_ready = true;
      this._init_toolbar();
    }
  }

  /**
   * Update the enabled/disabled state of navigation history toolbar buttons.
   *
   * @param fig - The Figure instance.
   * @param msg - Map of button names (e.g. ``'Back'``, ``'Forward'``) to boolean enabled state.
   */
  handle_history_buttons(fig: Figure, msg: Record<string, boolean>): void {
    for (const [key, enabled] of Object.entries(msg)) {
      const button = fig.buttons[key];
      if (!button) continue;
      button.disabled = !enabled;
      button.setAttribute('aria-disabled', String(!enabled));
    }
  }

  /**
   * Update visual state of Pan / Zoom toolbar buttons to reflect the active
   * navigation mode.
   *
   * @param fig - The Figure instance.
   * @param msg - Message containing ``mode`` (``'PAN'``, ``'ZOOM'``, or ``null``).
   */
  handle_navigate_mode(fig: Figure, msg: NavigateModeMessage): void {
    const mode = msg.mode;

    if (mode === 'PAN') {
      fig.buttons['Pan']?.classList.add('active');
      fig.buttons['Zoom']?.classList.remove('active');
    } else if (mode === 'ZOOM') {
      fig.buttons['Pan']?.classList.remove('active');
      fig.buttons['Zoom']?.classList.add('active');
    } else {
      fig.buttons['Pan']?.classList.remove('active');
      fig.buttons['Zoom']?.classList.remove('active');
    }
  }

  /**
   * Acknowledge receipt of a rendered image frame by sending an ``ack``
   * message to the server.
   *
   * Called automatically after each binary image is painted onto the canvas.
   */
  updated_canvas_event(): void {
    this.send_message('ack', {});
  }

  private _make_on_message_function(): (evt: MessageEvent) => void {
    return (evt: MessageEvent) => {
      // Handle binary image data (v0 protocol with 8-byte header)
      if (evt.data instanceof ArrayBuffer) {
        const { header, imageData } = parseBinaryImage(evt.data);

        // Update image mode based on header
        this.image_mode = header.typeMode === ImageTypeMode.FULL ? 'full' : 'diff';

        // Get correct MIME type from header
        const mimeType = getImageMimeType(header.format);

        // Create blob with correct type.
        // imageData is a Uint8Array *view* into the original ArrayBuffer
        // (offset past the 8-byte header), so we must NOT use .buffer here
        // — that would include the header bytes and corrupt the image.
        // The `as any` satisfies TS 5.9 which widens Uint8Array's backing
        // store to ArrayBufferLike (incompatible with BlobPart).
        const blob = new Blob([imageData as any], {
          type: mimeType,
        });

        // Free memory for previous frames
        if (this.imageObj.src) {
          URL.revokeObjectURL(this.imageObj.src);
        }

        this.imageObj.src = URL.createObjectURL(blob);
        this.updated_canvas_event();
        this.waiting = false;

        // Hide reconnect overlay on first image after reconnection
        if (this._reconnecting) {
          this._reconnecting = false;
          this._hideReconnectOverlay();
        }
        return;
      }

      // Handle legacy Blob format (for backward compatibility)
      if (evt.data instanceof Blob) {
        let img = evt.data;
        if (img.type !== 'image/png') {
          // Force PNG type
          img = new Blob([img], { type: 'image/png' });
        }

        // Free memory for previous frames
        if (this.imageObj.src) {
          URL.revokeObjectURL(this.imageObj.src);
        }

        this.imageObj.src = URL.createObjectURL(img);
        this.updated_canvas_event();
        this.waiting = false;
        return;
      } else if (
        typeof evt.data === 'string' &&
        evt.data.startsWith('data:image/png;base64')
      ) {
        this.imageObj.src = evt.data;
        this.updated_canvas_event();
        this.waiting = false;
        return;
      }

      const msg = JSON.parse(evt.data) as { type: string };
      const msg_type = msg.type;

      // Dynamic dispatch to handle_{type} methods.
      // This is a pattern inherited from matplotlib's JS layer.  The `any`
      // cast is unavoidable because the handler names are constructed at
      // runtime; individual handle_* methods carry their own typed signatures.
      const handler = `handle_${msg_type}` as keyof Figure;
      const callback = this[handler];

      if (typeof callback === 'function') {
        try {
          (callback as (fig: Figure, msg: unknown) => void).call(this, this, msg);
        } catch (e) {
          console.error(`Exception inside '${handler}' callback:`, e, msg);
        }
      }
    };
  }

  /**
   * Translate a DOM mouse event into a matplotlib event message and send it
   * to the server.
   *
   * @param event - The original DOM ``MouseEvent``.
   * @param name - matplotlib event name (e.g. ``'button_press'``, ``'motion_notify'``).
   * @returns Always ``false`` to prevent default browser behaviour.
   */
  mouse_event(event: MouseEvent, name: string): boolean {
    const canvas_pos = findpos(event);

    if (name === 'button_press') {
      this.canvas?.focus();
      this.canvas_div?.focus();
    }

    const x = canvas_pos.x * this.ratio;
    const y = canvas_pos.y * this.ratio;

    this.send_message(name, {
      x,
      y,
      button: event.button,
      step: (event as any).step,
      guiEvent: simpleKeys(event as any),
    });

    event.preventDefault();
    return false;
  }

  /**
   * Extension point for subclasses to handle additional key-event logic
   * before the event is forwarded to the server.
   *
   * @param _event - The original DOM ``KeyboardEvent``.
   * @param _name - matplotlib event name (``'key_press'`` or ``'key_release'``).
   */
  protected _key_event_extra(_event: KeyboardEvent, _name: string): void {
    // Hook for subclasses
  }

  /**
   * Translate a DOM keyboard event into a matplotlib key message and send it
   * to the server.
   *
   * Deduplicates key-repeat events for ``key_press`` and builds a modifier
   * prefix string (``ctrl+``, ``alt+``, ``shift+``) before the key value.
   *
   * @param event - The original DOM ``KeyboardEvent``.
   * @param name - matplotlib event name (``'key_press'`` or ``'key_release'``).
   * @returns Always ``false`` to prevent default browser behaviour.
   */
  key_event(event: KeyboardEvent, name: string): boolean {
    // Prevent repeat events
    if (name === 'key_press') {
      if (event.key === this._key) {
        return false;
      } else {
        this._key = event.key;
      }
    }
    if (name === 'key_release') {
      this._key = null;
    }

    let value = '';
    if (event.ctrlKey && event.key !== 'Control') {
      value += 'ctrl+';
    } else if (event.altKey && event.key !== 'Alt') {
      value += 'alt+';
    } else if (event.shiftKey && event.key !== 'Shift') {
      value += 'shift+';
    }

    value += 'k' + event.key;

    this._key_event_extra(event, name);

    this.send_message(name, { key: value, guiEvent: simpleKeys(event as any) });
    return false;
  }

  /**
   * Handle a toolbar button click.
   *
   * The ``'download'`` button delegates to ``handle_save()``; all other
   * buttons forward a ``toolbar_button`` message to the server.
   *
   * @param name - The matplotlib name of the toolbar button that was clicked.
   */
  toolbar_button_onclick(name: string): void {
    if (name === 'download') {
      this.handle_save(this, null);
    } else {
      this.send_message('toolbar_button', { name });
    }
  }

  /**
   * Display the tooltip text of a toolbar button in the status bar.
   *
   * @param tooltip - Tooltip string associated with the button being hovered.
   */
  toolbar_button_onmouseover(tooltip: string): void {
    if (this.message) {
      this.message.textContent = tooltip;
    }
  }

  // -- reconnection overlay ------------------------------------------------

  /**
   * Show a semi-transparent overlay with the given text.
   */
  private _showOverlay(text: string): void {
    this._ensureReconnectOverlay();
    if (this._reconnect_label) {
      this._reconnect_label.textContent = text;
    }
    if (this._reconnect_overlay) {
      this._reconnect_overlay.style.display = 'flex';
    }
  }

  /**
   * Hide the reconnection overlay.
   */
  private _hideReconnectOverlay(): void {
    if (this._reconnect_overlay) {
      this._reconnect_overlay.style.display = 'none';
    }
  }

  /**
   * Lazily create the overlay element inside the canvas_div.
   */
  private _ensureReconnectOverlay(): void {
    if (this._reconnect_overlay || !this.canvas_div) return;

    const overlay = document.createElement('div');
    overlay.style.cssText =
      'position: absolute;' +
      'inset: 0;' +
      'display: none;' +
      'align-items: center;' +
      'justify-content: center;' +
      'z-index: 10;' +
      'pointer-events: none;';

    const label = document.createElement('div');
    label.style.cssText =
      'background-color: rgba(0, 0, 0, 0.62);' +
      'color: white;' +
      'font-size: 16px;' +
      'padding: 12px 20px;' +
      'border-radius: 8px;';
    overlay.appendChild(label);

    this.canvas_div.appendChild(overlay);
    this._reconnect_overlay = overlay;
    this._reconnect_label = label;
  }

  /**
   * Clean up resources and remove DOM elements
   * Call this when the figure is no longer needed
   */
  destroy(): void {
    // Disconnect ResizeObserver to prevent memory leaks and runaway observers
    if (this.resizeObserverInstance) {
      this.resizeObserverInstance.disconnect();
      this.resizeObserverInstance = null;
    }

    // Close WebSocket if still open
    if (this.ws_manager) {
      this.ws_manager.close();
    }

    // Revoke any object URLs to free memory
    if (this.imageObj.src && this.imageObj.src.startsWith('blob:')) {
      URL.revokeObjectURL(this.imageObj.src);
    }

    // Remove DOM elements
    if (this.root && this.root.parentNode) {
      this.root.parentNode.removeChild(this.root);
    }
  }
}
