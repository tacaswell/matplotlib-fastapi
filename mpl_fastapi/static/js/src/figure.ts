/**
 * Figure - Matplotlib figure renderer with WebSocket communication
 *
 * This class handles:
 * - Canvas rendering and image updates
 * - Mouse and keyboard event handling
 * - Toolbar integration
 * - WebSocket message protocol
 */

import type { ImageMode } from './types.js';
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
  if (!targ && e.srcElement) {
    targ = e.srcElement as HTMLElement;
  }
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
  id: string;
  connection_id: string | null = null;
  ws_manager: WebSocketManager;
  ws: WebSocket | null = null; // Legacy property

  // Canvas and rendering
  canvas: HTMLCanvasElement | undefined;
  canvas_div: HTMLDivElement | undefined;
  context: CanvasRenderingContext2D | undefined;
  rubberband_canvas: HTMLCanvasElement | undefined;
  rubberband_context: CanvasRenderingContext2D | undefined;
  ratio: number = 1;
  image_mode: ImageMode = 'full';
  imageObj: HTMLImageElement;
  waiting: boolean = false;

  // UI elements
  root: HTMLDivElement;
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
  private ResizeObserver: any;
  private resizeObserverInstance: any;
  private _resize_canvas?: (width: number, height: number, forward: boolean) => void;

  // Download handler
  ondownload: (fig: Figure, format: string) => void;

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
      this.ws = this.ws_manager['ws'] as WebSocket; // Access private property
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

      // Send protocol version as first message from client (REQUIRED)
      this.send_message('protocol_version', { version: 0 });

      // Send initialization messages
      this.send_message('supports_binary', { value: this.supports_binary });
      this.send_message('send_image_mode', {});
      if (this.ratio !== 1) {
        this.send_message('set_device_pixel_ratio', {
          device_pixel_ratio: this.ratio,
        });
      }
      this.send_message('refresh', {});
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

  protected _canvas_extra_style(_canvas_div: HTMLDivElement): void {
    // Hook for subclasses
  }

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

    const backingStore =
      (context as any).backingStorePixelRatio ||
      (context as any).webkitBackingStorePixelRatio ||
      (context as any).mozBackingStorePixelRatio ||
      (context as any).msBackingStorePixelRatio ||
      (context as any).oBackingStorePixelRatio ||
      (context as any).backingStorePixelRatio ||
      1;

    this.ratio = (window.devicePixelRatio || 1) / backingStore;

    const rubberband_canvas = (this.rubberband_canvas =
      document.createElement('canvas'));
    rubberband_canvas.setAttribute(
      'style',
      'box-sizing: content-box; position: absolute; left: 0; top: 0; z-index: 1;'
    );

    // Apply a ponyfill if ResizeObserver is not implemented by browser
    this.ResizeObserver =
      window.ResizeObserver ||
      (window as any)._JSXTOOLS_RESIZE_OBSERVER?.({}).ResizeObserver;

    if (!this.ResizeObserver) {
      console.warn('ResizeObserver not available');
      return;
    }

    this.resizeObserverInstance = new this.ResizeObserver((entries: any[]) => {
      for (const entry of entries) {
        let width: number, height: number;

        if (entry.contentBoxSize) {
          if (entry.contentBoxSize instanceof Array) {
            // Chrome 84+
            width = entry.contentBoxSize[0].inlineSize;
            height = entry.contentBoxSize[0].blockSize;
          } else {
            // Firefox
            width = entry.contentBoxSize.inlineSize;
            height = entry.contentBoxSize.blockSize;
          }
        } else {
          // Chrome <84
          width = entry.contentRect.width;
          height = entry.contentRect.height;
        }

        // Keep canvas and rubberband canvas in sync
        if (entry.devicePixelContentBoxSize) {
          // Chrome 84+
          canvas.setAttribute(
            'width',
            String(entry.devicePixelContentBoxSize[0].inlineSize)
          );
          canvas.setAttribute(
            'height',
            String(entry.devicePixelContentBoxSize[0].blockSize)
          );
        } else {
          canvas.setAttribute('width', String(width * this.ratio));
          canvas.setAttribute('height', String(height * this.ratio));
        }
        canvas.setAttribute('style', `width: ${width}px; height: ${height}px;`);

        rubberband_canvas.setAttribute('width', String(width));
        rubberband_canvas.setAttribute('height', String(height));

        // Update size in Python (ignore initial 0/0 size)
        if (this.ws && this.ws.readyState === 1 && width !== 0 && height !== 0) {
          this.request_resize(width, height);
        }
      }
    });
    this.resizeObserverInstance.observe(canvas_div);

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

  request_resize(x_pixels: number, y_pixels: number): void {
    this.send_message('resize', { width: x_pixels, height: y_pixels });
  }

  send_message(type: string, properties: Record<string, any>): void {
    properties['type'] = type;
    properties['figure_id'] = this.id;
    this.ws_manager.send(JSON.stringify(properties));
  }

  send_draw_message(): void {
    if (!this.waiting) {
      this.waiting = true;
      this.ws_manager.send(JSON.stringify({ type: 'draw', figure_id: this.id }));
    }
  }

  send_update(form_id: string): void {
    const form = document.getElementById(form_id);
    if (!form) {
      console.error('Update form not found:', form_id);
      return;
    }

    const params: Record<string, any> = {};
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach((input) => {
      const inputEl = input as HTMLInputElement | HTMLSelectElement;
      let value: any = inputEl.value;

      if ((inputEl as HTMLInputElement).type === 'number') {
        value = parseFloat(value);
      }
      params[inputEl.name] = value;
    });

    this.send_message('update_params', { params });
  }

  handle_save(fig: Figure, _msg: any): void {
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

  handle_save_complete(fig: Figure, msg: any): void {
    const download_url = msg['download_url'];
    const filename = msg['filename'];

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

  handle_save_error(fig: Figure, msg: any): void {
    const error_message = msg['message'];
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

  handle_resize(fig: Figure, msg: any): void {
    const size = msg['size'];
    if (!fig.canvas || !fig._resize_canvas) return;

    if (size[0] !== fig.canvas.width || size[1] !== fig.canvas.height) {
      fig._resize_canvas(size[0], size[1], msg['forward']);
      fig.send_message('refresh', {});
    }
  }

  handle_rubberband(fig: Figure, msg: any): void {
    if (!fig.canvas || !fig.rubberband_context) return;

    let x0 = msg['x0'] / fig.ratio;
    let y0 = (fig.canvas.height - msg['y0']) / fig.ratio;
    let x1 = msg['x1'] / fig.ratio;
    let y1 = (fig.canvas.height - msg['y1']) / fig.ratio;

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

  handle_figure_label(fig: Figure, msg: any): void {
    if (fig.header) {
      fig.header.textContent = msg['label'];
    }
  }

  handle_cursor(fig: Figure, msg: any): void {
    if (fig.rubberband_canvas) {
      fig.rubberband_canvas.style.cursor = msg['cursor'];
    }
  }

  handle_message(fig: Figure, msg: any): void {
    if (fig.message) {
      fig.message.textContent = msg['message'];
    }
  }

  handle_protocol_version(fig: Figure, msg: any): void {
    const server_version = msg['version'];
    // Protocol version is REQUIRED
    if (server_version == null) {
      console.error('Protocol version missing from server message');
      if (fig.ws_manager) {
        fig.ws_manager.close();
      }
      throw new Error('Protocol version is required');
    }
    if (server_version !== 0) {
      console.error(
        `Protocol version mismatch: client expects 0, server sent ${server_version}`
      );
      if (fig.ws_manager) {
        fig.ws_manager.close();
      }
      throw new Error(
        `Incompatible protocol version. Client expects 0, got ${server_version}`
      );
    }
    console.log(`Server protocol version validated: ${server_version}`);
  }

  handle_draw(fig: Figure, _msg: any): void {
    fig.send_draw_message();
  }

  handle_image_mode(fig: Figure, msg: any): void {
    fig.image_mode = msg['mode'];
  }

  handle_connection_id(fig: Figure, msg: any): void {
    fig.connection_id = msg['id'];
  }

  handle_toolbar_config(fig: Figure, msg: any): void {
    fig.toolbar_items = msg['items'];
    fig._check_toolbar_ready();
  }

  handle_save_formats(fig: Figure, msg: any): void {
    fig.save_formats = msg['formats'];
    fig._check_toolbar_ready();
  }

  handle_default_save_format(fig: Figure, msg: any): void {
    fig.default_save_format = msg['format'];
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

  handle_history_buttons(fig: Figure, msg: any): void {
    for (const key in msg) {
      if (!(key in fig.buttons)) {
        continue;
      }
      const button = fig.buttons[key];
      if (!button) continue;
      button.disabled = !msg[key];
      button.setAttribute('aria-disabled', String(!msg[key]));
    }
  }

  handle_navigate_mode(fig: Figure, msg: any): void {
    const mode = msg['mode'];

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

  updated_canvas_event(): void {
    this.send_message('ack', {});
  }

  private _make_on_message_function(): (evt: MessageEvent) => void {
    return (evt: MessageEvent) => {
      if (evt.data instanceof Blob) {
        let img = evt.data;
        if (img.type !== 'image/png') {
          // Force PNG type
          img = new Blob([img], { type: 'image/png' });
        }

        // Free memory for previous frames
        if (this.imageObj.src) {
          (window.URL || (window as any).webkitURL).revokeObjectURL(this.imageObj.src);
        }

        this.imageObj.src = (window.URL || (window as any).webkitURL).createObjectURL(
          img
        );
        this.updated_canvas_event();
        this.waiting = false;
        return;
      } else if (
        typeof evt.data === 'string' &&
        evt.data.slice(0, 21) === 'data:image/png;base64'
      ) {
        this.imageObj.src = evt.data;
        this.updated_canvas_event();
        this.waiting = false;
        return;
      }

      const msg = JSON.parse(evt.data);
      const msg_type = msg['type'];

      // Call the handle_{type} callback
      const callback = (this as any)['handle_' + msg_type];

      if (callback) {
        try {
          callback.call(this, this, msg);
        } catch (e) {
          console.log(`Exception inside 'handle_${msg_type}' callback:`, e, msg);
        }
      }
    };
  }

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

  protected _key_event_extra(_event: KeyboardEvent, _name: string): void {
    // Hook for subclasses
  }

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

  toolbar_button_onclick(name: string): void {
    if (name === 'download') {
      this.handle_save(this, null);
    } else {
      this.send_message('toolbar_button', { name });
    }
  }

  toolbar_button_onmouseover(tooltip: string): void {
    if (this.message) {
      this.message.textContent = tooltip;
    }
  }
}

// Export Figure to global window.mpl namespace for backwards compatibility
if (typeof window !== 'undefined') {
  if (!window.mpl) {
    window.mpl = {} as any;
  }
  window.mpl.Figure = Figure as any;
}
