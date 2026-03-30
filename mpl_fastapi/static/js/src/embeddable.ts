/**
 * MatplotlibEmbeddable - Unified matplotlib component
 *
 * A composable, framework-agnostic matplotlib component that provides a clean
 * JavaScript API for embedding interactive matplotlib figures into any web
 * application without requiring server-rendered HTML templates.
 *
 * @example
 * ```typescript
 * const plot = new MatplotlibEmbeddable({
 *   container: document.getElementById('my-plot'),
 *   plotName: 'sine',
 *   baseUrl: '/plots',
 *   initParams: { frequency: 2.0, amplitude: 1.0 },
 *   updateParams: { phase: 0.0 },
 *   onConnect: () => console.log('Connected'),
 *   onError: (error) => console.error('Error:', error),
 *   onUpdate: () => console.log('Plot updated'),
 *   showToolbar: true,
 *   showUpdateForm: true,
 *   staticPath: '/mpl-static'
 * });
 *
 * // Later: update parameters
 * plot.update({ phase: 1.57 });
 *
 * // Disconnect and cleanup
 * plot.disconnect();
 * ```
 */

import type { EmbeddableConfig, PlotSchemaResponse } from './types.js';
import { WebSocketManager } from './websocket-manager.js';
import { Figure } from './figure.js';

/**
 * Internal resolved config type.
 *
 * All fields are required except `token` which stays optional since
 * not every deployment uses authentication.
 */
type ResolvedConfig = Required<Omit<EmbeddableConfig, 'token'>> & { token: string | undefined };

/**
 * Main embeddable matplotlib component class
 *
 * This provides a unified API for embedding interactive matplotlib figures
 * into any web application, supporting both template-based and programmatic usage.
 */
export class MatplotlibEmbeddable {
  private readonly config: ResolvedConfig;
  private ws_manager: WebSocketManager | null = null;
  private figure: Figure | null = null;
  private connected: boolean = false;
  private updateSchema: PlotSchemaResponse['update_schema'] | null = null;
  private submitButton: HTMLButtonElement | null = null;
  private readonly instanceId: string;

  constructor(config: EmbeddableConfig) {
    // Generate unique instance ID to avoid ID collisions when multiple plots on same page
    this.instanceId = `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;

    // Validate required options
    if (!config.container) {
      throw new Error('container option is required');
    }
    if (!config.plotName) {
      throw new Error('plotName option is required');
    }

    // Set defaults for optional properties
    this.config = {
      container: config.container,
      plotName: config.plotName,
      baseUrl: config.baseUrl ?? '',
      initParams: config.initParams ?? {},
      updateParams: config.updateParams ?? {},
      token: config.token,
      onConnect: config.onConnect ?? (() => {}),
      onError: config.onError ?? ((err: Error) => console.error('MPL Error:', err)),
      onUpdate: config.onUpdate ?? (() => {}),
      onDisconnect: config.onDisconnect ?? (() => {}),
      showToolbar: config.showToolbar ?? true,
      showUpdateForm: config.showUpdateForm ?? true,
      staticPath: config.staticPath ?? '/mpl-static',
      autoConnect: config.autoConnect ?? true,
      reconnect: config.reconnect ?? {},
      onReconnecting: config.onReconnecting ?? (() => {}),
      onReconnected: config.onReconnected ?? (() => {}),
      onReconnectFailed: config.onReconnectFailed ?? (() => {}),
    };

    // Set static path globally for Figure class compatibility
    if (typeof window !== 'undefined') {
      window.MPL_STATIC_PATH = this.config.staticPath;
    }

    // Inject required CSS if not already present
    this._injectCSS();

    // Auto-connect if requested
    if (this.config.autoConnect) {
      this.connect();
    }
  }

  /**
   * Inject required matplotlib CSS files
   */
  private _injectCSS(): void {
    const cssFiles = ['boilerplate.css', 'fbm.css', 'mpl.css'];
    const head = document.head || document.getElementsByTagName('head')[0];

    cssFiles.forEach((file) => {
      const linkId = `mpl-css-${file}`;
      // Check if already injected
      if (!document.getElementById(linkId)) {
        const link = document.createElement('link');
        link.id = linkId;
        link.rel = 'stylesheet';
        link.type = 'text/css';
        link.href = `${this.config.staticPath}/css/${file}`;
        head.appendChild(link);
      }
    });
  }

  /**
   * Construct WebSocket URL with query parameters
   */
  private _buildWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    // Use v0 protocol endpoint
    let wsUrl = `${protocol}//${host}${this.config.baseUrl}/ws/v0/${this.config.plotName}`;

    // Add init parameters as query string
    const params = new URLSearchParams(
      Object.entries(this.config.initParams).map(([k, v]) => [k, String(v)])
    );

    // Add update parameters with _update. prefix
    for (const [k, v] of Object.entries(this.config.updateParams)) {
      params.set(`_update.${k}`, String(v));
    }

    // Add auth token if configured
    if (this.config.token) {
      params.set('token', this.config.token);
    }

    const queryString = params.toString();
    if (queryString) {
      wsUrl += '?' + queryString;
    }

    return wsUrl;
  }

  /**
   * Connect to the WebSocket and initialize the plot
   */
  async connect(): Promise<void> {
    if (this.connected) {
      console.warn('Already connected');
      return;
    }

    try {
      // Fetch update schema if showUpdateForm is true
      if (this.config.showUpdateForm) {
        await this._fetchUpdateSchema();
      }

      // Create WebSocketManager
      const wsUrl = this._buildWebSocketUrl();
      this.ws_manager = new WebSocketManager(wsUrl, this.config.reconnect);

      // Set up connection handlers
      this.ws_manager.onOpen(() => {
        this.connected = true;
        // Enable submit button if it exists
        if (this.submitButton) {
          this.submitButton.disabled = false;
          this.submitButton.style.opacity = '1';
          this.submitButton.style.cursor = 'pointer';
        }
        this.config.onConnect();
      });

      this.ws_manager.onClose((event: CloseEvent) => {
        if (this.connected) {
          this.connected = false;
          // Disable submit button if it exists
          if (this.submitButton) {
            this.submitButton.disabled = true;
            this.submitButton.style.opacity = '0.5';
            this.submitButton.style.cursor = 'not-allowed';
          }
          this.config.onDisconnect();
        } else if (event.code !== 1000 && !this.ws_manager?.isReconnecting()) {
          // Connection failed before it was established AND we are not in
          // a reconnect cycle.  Clean up to prevent resource leaks.
          this._cleanupFigure();
          this.config.onError(
            new Error(`Connection failed: ${event.reason || 'Unknown error'}`)
          );
        }
      });

      this.ws_manager.onError(() => {
        // During reconnection the browser fires onerror for every failed
        // attempt.  Only clean up if we're NOT in a reconnect cycle;
        // otherwise let the reconnect machinery keep trying.
        if (!this.ws_manager?.isReconnecting()) {
          this._cleanupFigure();
          this.config.onError(new Error('WebSocket connection failed'));
        }
      });

      // Reconnection handlers — forward to config callbacks
      this.ws_manager.onReconnecting(this.config.onReconnecting);

      this.ws_manager.onReconnected(() => {
        this.connected = true;
        // Re-enable submit button
        if (this.submitButton) {
          this.submitButton.disabled = false;
          this.submitButton.style.opacity = '1';
          this.submitButton.style.cursor = 'pointer';
        }
        this.config.onReconnected();
      });

      this.ws_manager.onReconnectFailed(this.config.onReconnectFailed);

      // Initialize figure first (before connecting), it will register its handlers
      this._initializeFigure();

      // Now connect the WebSocket
      this.ws_manager.connect();
    } catch (error) {
      this.config.onError(error as Error);
    }
  }

  /**
   * Fetch the update parameter schema for this plot
   */
  private async _fetchUpdateSchema(): Promise<void> {
    try {
      const headers: Record<string, string> = {};
      if (this.config.token) {
        headers['Authorization'] = `Bearer ${this.config.token}`;
      }
      const response = await fetch(
        `${this.config.baseUrl}/api/plots/${this.config.plotName}/schema`,
        { headers }
      );
      if (!response.ok) {
        console.warn('Could not fetch update schema');
        return;
      }
      const data: PlotSchemaResponse = await response.json();
      this.updateSchema = data.update_schema;
    } catch (error) {
      console.warn('Error fetching update schema:', error);
    }
  }

  /**
   * Initialize the matplotlib figure in the container
   */
  private _initializeFigure(): void {
    // Clear container
    this.config.container.innerHTML = '';

    // Create wrapper div for the figure
    const wrapper = document.createElement('div');
    wrapper.className = 'matplotlib-embeddable-wrapper';

    // Create figure container - keep it minimal
    const figureContainer = document.createElement('div');
    figureContainer.className = 'matplotlib-figure-container';
    figureContainer.id = `mpl-figure-${this.config.plotName}-${Date.now()}`;
    wrapper.appendChild(figureContainer);

    // Create update form if schema is available
    if (this.config.showUpdateForm && this.updateSchema) {
      const updateForm = this._createUpdateForm();
      updateForm.style.marginTop = '10px';
      wrapper.appendChild(updateForm);
    }

    this.config.container.appendChild(wrapper);

    // Initialize the Figure
    if (!this.ws_manager) {
      this.config.onError(new Error('WebSocketManager not initialized'));
      return;
    }

    this.figure = new Figure(this.config.plotName, this.ws_manager, figureContainer);

    // Ensure focus is set after a short delay (critical for keyboard events)
    setTimeout(() => {
      if (this.figure && this.figure.canvas_div) {
        this.figure.canvas_div.focus();
      }
    }, 150);
  }

  /**
   * Create an update form from the schema
   */
  private _createUpdateForm(): HTMLDivElement {
    const formContainer = document.createElement('div');
    formContainer.className = 'matplotlib-update-form';
    formContainer.style.padding = '10px';
    formContainer.style.border = '1px solid #ccc';
    formContainer.style.borderRadius = '4px';
    formContainer.style.backgroundColor = '#f9f9f9';

    const title = document.createElement('h4');
    title.textContent = 'Update Parameters';
    title.style.margin = '0 0 10px 0';
    formContainer.appendChild(title);

    const form = document.createElement('form');
    form.id = `update-form-${this.instanceId}`;
    form.onsubmit = (e) => {
      e.preventDefault();
      this._handleUpdateFormSubmit();
    };

    // Create inputs from schema
    if (this.updateSchema) {
      const properties = this.updateSchema.properties || {};
      for (const [paramName, paramInfo] of Object.entries(properties)) {
        const fieldDiv = document.createElement('div');
        fieldDiv.style.marginBottom = '10px';

        const label = document.createElement('label');
        label.textContent = paramInfo.title || paramName;
        label.style.display = 'block';
        label.style.marginBottom = '4px';
        label.style.fontWeight = 'bold';

        if (paramInfo.description) {
          const desc = document.createElement('span');
          desc.textContent = ` (${paramInfo.description})`;
          desc.style.fontWeight = 'normal';
          desc.style.fontSize = '0.9em';
          desc.style.color = '#666';
          label.appendChild(desc);
        }

        const input = document.createElement('input');
        input.type = 'number';
        input.name = paramName;
        input.id = `${paramName}-${this.instanceId}`;
        input.step = 'any';
        input.style.width = '100%';
        input.style.padding = '4px';

        // Set default value
        if (this.config.updateParams[paramName] !== undefined) {
          input.value = String(this.config.updateParams[paramName]);
        } else if (paramInfo.default !== undefined) {
          input.value = String(paramInfo.default);
        }

        // Set min/max if available
        if (paramInfo.minimum !== undefined) {
          input.min = String(paramInfo.minimum);
        }
        if (paramInfo.maximum !== undefined) {
          input.max = String(paramInfo.maximum);
        }

        fieldDiv.appendChild(label);
        fieldDiv.appendChild(input);
        form.appendChild(fieldDiv);
      }
    }

    const submitButton = document.createElement('button');
    submitButton.type = 'submit';
    submitButton.textContent = 'Update Plot';
    submitButton.style.padding = '8px 16px';
    submitButton.style.backgroundColor = '#007bff';
    submitButton.style.color = 'white';
    submitButton.style.border = 'none';
    submitButton.style.borderRadius = '4px';
    submitButton.style.cursor = 'pointer';
    submitButton.style.marginTop = '10px';

    // Disable button initially until connected
    submitButton.disabled = true;
    submitButton.style.opacity = '0.5';
    submitButton.style.cursor = 'not-allowed';

    submitButton.onmouseover = () => {
      if (!submitButton.disabled) {
        submitButton.style.backgroundColor = '#0056b3';
      }
    };
    submitButton.onmouseout = () => {
      if (!submitButton.disabled) {
        submitButton.style.backgroundColor = '#007bff';
      }
    };

    form.appendChild(submitButton);
    formContainer.appendChild(form);

    // Store reference to submit button
    this.submitButton = submitButton;

    return formContainer;
  }

  /**
   * Handle update form submission
   */
  private _handleUpdateFormSubmit(): void {
    const formId = `update-form-${this.instanceId}`;
    const form = document.getElementById(formId);
    if (!form) {
      console.error('Update form not found');
      return;
    }

    const params: Record<string, unknown> = {};
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach((input) => {
      const inputEl = input as HTMLInputElement | HTMLSelectElement;
      let value: unknown = inputEl.value;

      // Convert to appropriate type
      if ((inputEl as HTMLInputElement).type === 'number') {
        value = parseFloat(inputEl.value);
      } else if ((inputEl as HTMLInputElement).type === 'checkbox') {
        value = (inputEl as HTMLInputElement).checked;
      }
      params[inputEl.name] = value;
    });

    this.update(params);
  }

  /**
   * Update plot parameters
   *
   * @param params - Parameters to update
   */
  update(params: Record<string, unknown>): void {
    if (!this.connected) {
      this.config.onError(new Error('Not connected. Call connect() first.'));
      return;
    }

    if (!this.figure) {
      this.config.onError(new Error('Figure not initialized'));
      return;
    }

    // Update internal state
    this.config.updateParams = { ...this.config.updateParams, ...params };

    // Send update message via WebSocket
    this.figure.send_message('update_params', { params });

    // Trigger callback
    this.config.onUpdate(params);
  }

  /**
   * Clean up the figure without closing WebSocket
   * Used internally when connection fails
   */
  private _cleanupFigure(): void {
    if (this.figure) {
      this.figure.destroy();
      this.figure = null;
    }
  }

  /**
   * Disconnect and cleanup
   */
  disconnect(): void {
    if (this.ws_manager) {
      this.ws_manager.close();
      this.ws_manager = null;
    }
    this._cleanupFigure();
    this.connected = false;
  }

  /**
   * Manually focus the plot canvas (useful for keyboard events)
   */
  focus(): void {
    if (this.figure && this.figure.canvas_div) {
      this.figure.canvas_div.focus();
    }
  }

  /**
   * Check if currently connected
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Get current update parameters
   */
  getUpdateParams(): Record<string, unknown> {
    return { ...this.config.updateParams };
  }

  /**
   * Get initial parameters
   */
  getInitParams(): Record<string, unknown> {
    return { ...this.config.initParams };
  }
}
