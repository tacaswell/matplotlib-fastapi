/**
 * MatplotlibEmbeddable - A composable, framework-agnostic matplotlib component
 *
 * This provides a clean JavaScript API for embedding interactive matplotlib figures
 * into any web application without requiring server-rendered HTML templates.
 *
 * Usage:
 *   const plot = new MatplotlibEmbeddable({
 *     container: document.getElementById('my-plot'),
 *     plotName: 'sine',
 *     baseUrl: '/plots',
 *     initParams: { frequency: 2.0, amplitude: 1.0 },
 *     updateParams: { phase: 0.0 },
 *     onConnect: () => console.log('Connected'),
 *     onError: (error) => console.error('Error:', error),
 *     onUpdate: () => console.log('Plot updated'),
 *     showToolbar: true,
 *     showUpdateForm: true,
 *     staticPath: '/mpl-static'
 *   });
 *
 *   // Later: update parameters
 *   plot.update({ phase: 1.57 });
 *
 *   // Disconnect and cleanup
 *   plot.disconnect();
 */

(function(window) {
    'use strict';

    // Reuse utilities from existing mpl namespace
    const mpl = window.mpl || {};

    /**
     * Get the appropriate WebSocket constructor for the browser
     */
    function getWebSocketType() {
        if (typeof WebSocket !== 'undefined') {
            return WebSocket;
        } else if (typeof MozWebSocket !== 'undefined') {
            return MozWebSocket;
        } else {
            throw new Error(
                'Your browser does not have WebSocket support. ' +
                'Please try Chrome, Safari or Firefox ≥ 6.'
            );
        }
    }

    /**
     * Main embeddable matplotlib component class
     */
    class MatplotlibEmbeddable {
        constructor(options) {
            // Validate required options
            if (!options.container) {
                throw new Error('container option is required');
            }
            if (!options.plotName) {
                throw new Error('plotName option is required');
            }

            // Store options with defaults
            this.container = options.container;
            this.plotName = options.plotName;
            this.baseUrl = options.baseUrl || '';
            this.initParams = options.initParams || {};
            this.updateParams = options.updateParams || {};
            this.onConnect = options.onConnect || (() => {});
            this.onError = options.onError || ((err) => console.error('MPL Error:', err));
            this.onUpdate = options.onUpdate || (() => {});
            this.onDisconnect = options.onDisconnect || (() => {});
            this.showToolbar = options.showToolbar !== false; // default true
            this.showUpdateForm = options.showUpdateForm !== false; // default true
            this.staticPath = options.staticPath || '/mpl-static';
            this.autoConnect = options.autoConnect !== false; // default true

            // Internal state
            this.ws = null;
            this.figure = null;
            this.connected = false;
            this.updateSchema = null;
            this.submitButton = null;
            this._connectionCheckInterval = null;

            // Set static path globally for mpl.js compatibility
            window.MPL_STATIC_PATH = this.staticPath;

            // Inject required CSS if not already present
            this._injectCSS();

            // Auto-connect if requested
            if (this.autoConnect) {
                this.connect();
            }
        }

        /**
         * Inject required matplotlib CSS files
         */
        _injectCSS() {
            const cssFiles = ['boilerplate.css', 'fbm.css', 'mpl.css'];
            const head = document.head || document.getElementsByTagName('head')[0];

            cssFiles.forEach(file => {
                const linkId = `mpl-css-${file}`;
                // Check if already injected
                if (!document.getElementById(linkId)) {
                    const link = document.createElement('link');
                    link.id = linkId;
                    link.rel = 'stylesheet';
                    link.type = 'text/css';
                    link.href = `${this.staticPath}/css/${file}`;
                    head.appendChild(link);
                }
            });
        }

        /**
         * Construct WebSocket URL with query parameters
         */
        _buildWebSocketUrl() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            let wsUrl = `${protocol}//${host}${this.baseUrl}/ws/${this.plotName}`;

            // Add init parameters as query string
            const params = new URLSearchParams(this.initParams);
            const queryString = params.toString();
            if (queryString) {
                wsUrl += '?' + queryString;
            }

            return wsUrl;
        }

        /**
         * Connect to the WebSocket and initialize the plot
         */
        async connect() {
            if (this.connected) {
                console.warn('Already connected');
                return;
            }

            try {
                // Fetch update schema if showUpdateForm is true
                if (this.showUpdateForm) {
                    await this._fetchUpdateSchema();
                }

                // Create WebSocketManager
                const wsUrl = this._buildWebSocketUrl();
                this.ws_manager = new window.mpl.WebSocketManager(wsUrl);

                // Set up connection handlers
                this.ws_manager.onOpen(() => {
                    this.connected = true;
                    // Enable submit button if it exists
                    if (this.submitButton) {
                        this.submitButton.disabled = false;
                        this.submitButton.style.opacity = '1';
                        this.submitButton.style.cursor = 'pointer';
                    }
                    this.onConnect();
                });

                this.ws_manager.onClose((event) => {
                    if (this.connected) {
                        this.connected = false;
                        // Disable submit button if it exists
                        if (this.submitButton) {
                            this.submitButton.disabled = true;
                            this.submitButton.style.opacity = '0.5';
                            this.submitButton.style.cursor = 'not-allowed';
                        }
                        this.onDisconnect();
                    } else if (event.code !== 1000) {
                        // Connection failed before it was established
                        this.onError(new Error(`Connection failed: ${event.reason || 'Unknown error'}`));
                    }
                });

                this.ws_manager.onError((event) => {
                    this.onError(new Error('WebSocket connection failed'));
                });

                // Initialize figure first (before connecting), it will register its handlers
                this._initializeFigure();

                // Now connect the WebSocket
                this.ws_manager.connect();

            } catch (error) {
                this.onError(error);
            }
        }

        /**
         * Fetch the update parameter schema for this plot
         */
        async _fetchUpdateSchema() {
            try {
                const response = await fetch(`${this.baseUrl}/api/plots/${this.plotName}/schema`);
                if (!response.ok) {
                    console.warn('Could not fetch update schema');
                    return;
                }
                const data = await response.json();
                this.updateSchema = data.update_schema;
            } catch (error) {
                console.warn('Error fetching update schema:', error);
            }
        }

        /**
         * Initialize the matplotlib figure in the container
         */
        _initializeFigure() {
            // Clear container
            this.container.innerHTML = '';

            // Create wrapper div for the figure
            const wrapper = document.createElement('div');
            wrapper.className = 'matplotlib-embeddable-wrapper';

            // Create figure container - keep it minimal
            const figureContainer = document.createElement('div');
            figureContainer.className = 'matplotlib-figure-container';
            figureContainer.id = `mpl-figure-${this.plotName}-${Date.now()}`;
            wrapper.appendChild(figureContainer);

            // Create update form if schema is available
            if (this.showUpdateForm && this.updateSchema) {
                const updateForm = this._createUpdateForm();
                updateForm.style.marginTop = '10px';
                wrapper.appendChild(updateForm);
            }

            this.container.appendChild(wrapper);

            // Initialize the mpl.figure
            // Pass the WebSocketManager to mpl.figure
            if (window.mpl && window.mpl.figure) {
                this.figure = new window.mpl.figure(
                    this.plotName,
                    this.ws_manager,
                    figureContainer
                );

                // Store reference to this instance in the figure
                this.figure._embeddableInstance = this;

                // Ensure focus is set after a short delay (mpl.js also does this)
                // This is critical for keyboard events to work
                window.setTimeout(() => {
                    if (this.figure && this.figure.canvas_div) {
                        this.figure.canvas_div.focus();
                    }
                }, 150);
            } else {
                this.onError(new Error('mpl.figure not found. Ensure mpl.js is loaded.'));
            }
        }

        /**
         * Create an update form from the schema
         */
        _createUpdateForm() {
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
            form.id = `update-form-${this.plotName}`;
            form.onsubmit = (e) => {
                e.preventDefault();
                this._handleUpdateFormSubmit();
            };

            // Create inputs from schema
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
                input.id = `${paramName}-${this.plotName}`;
                input.step = 'any';
                input.style.width = '100%';
                input.style.padding = '4px';

                // Set default value
                if (this.updateParams[paramName] !== undefined) {
                    input.value = this.updateParams[paramName];
                } else if (paramInfo.default !== undefined) {
                    input.value = paramInfo.default;
                }

                // Set min/max if available
                if (paramInfo.minimum !== undefined) {
                    input.min = paramInfo.minimum;
                }
                if (paramInfo.maximum !== undefined) {
                    input.max = paramInfo.maximum;
                }

                fieldDiv.appendChild(label);
                fieldDiv.appendChild(input);
                form.appendChild(fieldDiv);
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
        _handleUpdateFormSubmit() {
            const form = document.getElementById(`update-form-${this.plotName}`);
            if (!form) {
                console.error('Update form not found');
                return;
            }

            const params = {};
            const inputs = form.querySelectorAll('input, select');
            inputs.forEach((input) => {
                let value = input.value;
                // Convert to appropriate type
                if (input.type === 'number') {
                    value = parseFloat(value);
                } else if (input.type === 'checkbox') {
                    value = input.checked;
                }
                params[input.name] = value;
            });

            this.update(params);
        }

        /**
         * Update plot parameters
         * @param {Object} params - Parameters to update
         */
        update(params) {
            if (!this.connected) {
                this.onError(new Error('Not connected. Call connect() first.'));
                return;
            }

            if (!this.figure) {
                this.onError(new Error('Figure not initialized'));
                return;
            }

            // Update internal state
            this.updateParams = { ...this.updateParams, ...params };

            // Send update message via WebSocket
            this.figure.send_message('update_params', { params: params });

            // Trigger callback
            this.onUpdate(params);
        }

        /**
         * Disconnect and cleanup
         */
        disconnect() {
            if (this.ws_manager) {
                this.ws_manager.close();
                this.ws_manager = null;
            }
            if (this.figure && this.figure.root) {
                this.figure.root.remove();
            }
            this.connected = false;
            this.figure = null;
        }

        /**
         * Manually focus the plot canvas (useful for keyboard events)
         */
        focus() {
            if (this.figure && this.figure.canvas_div) {
                this.figure.canvas_div.focus();
            }
        }

        /**
         * Check if currently connected
         */
        isConnected() {
            return this.connected;
        }

        /**
         * Get current update parameters
         */
        getUpdateParams() {
            return { ...this.updateParams };
        }

        /**
         * Get current init parameters
         */
        getInitParams() {
            return { ...this.initParams };
        }
    }

    // Export to global namespace
    window.MatplotlibEmbeddable = MatplotlibEmbeddable;

    // Also export as module if module system is available
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = MatplotlibEmbeddable;
    }

})(window);
