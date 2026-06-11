/**
 * WebSocketManager - Manages WebSocket connection lifecycle with reconnection
 *
 * This class ensures proper ordering of connection and message handling by
 * allowing handlers to be registered before the connection is established.
 * This prevents race conditions where messages might arrive before handlers
 * are ready.
 *
 * Automatic reconnection with exponential backoff is built in.  When the
 * WebSocket closes unexpectedly (i.e. not via an explicit ``close()`` call),
 * the manager will attempt to reconnect up to ``maxReconnectAttempts`` times
 * using exponential backoff with jitter.  Progress is reported through
 * ``onReconnecting`` / ``onReconnected`` handlers so the UI can show an
 * overlay.
 */

/**
 * Configuration for reconnection behaviour.
 */
export interface ReconnectConfig {
  /** Maximum number of reconnection attempts before giving up (0 = disabled). */
  maxAttempts?: number;
  /** Initial backoff delay in milliseconds. */
  initialDelayMs?: number;
  /** Maximum backoff delay cap in milliseconds. */
  maxDelayMs?: number;
  /** Exponential base for backoff growth. */
  backoffBase?: number;
}

const DEFAULT_RECONNECT: Required<ReconnectConfig> = {
  maxAttempts: 10,
  initialDelayMs: 500,
  maxDelayMs: 15_000,
  backoffBase: 2.0,
};

export class WebSocketManager {
  private readonly url: string;
  private ws: WebSocket | null;
  private readonly messageHandlers: Array<(event: MessageEvent) => void>;
  private readonly openHandlers: Array<(event: Event) => void>;
  private readonly closeHandlers: Array<(event: CloseEvent) => void>;
  private readonly errorHandlers: Array<(event: Event) => void>;

  // Reconnection handlers
  private readonly reconnectingHandlers: Array<
    (attempt: number, maxAttempts: number) => void
  >;
  private readonly reconnectedHandlers: Array<() => void>;
  private readonly reconnectFailedHandlers: Array<() => void>;

  private _isConnecting: boolean;
  private _isConnected: boolean;

  // Explicit-close flag — suppresses reconnection
  private _explicitClose: boolean;

  // Reconnection state
  private readonly _reconnectConfig: Required<ReconnectConfig>;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null;
  private _reconnectAttempt: number;

  /**
   * Create a new WebSocketManager.
   *
   * @param url - Full WebSocket URL to connect to (e.g. ``wss://host/ws/v0/plot``).
   * @param reconnect - Optional reconnection configuration overrides.
   */
  constructor(url: string, reconnect?: ReconnectConfig) {
    this.url = url;
    this.ws = null;
    this.messageHandlers = [];
    this.openHandlers = [];
    this.closeHandlers = [];
    this.errorHandlers = [];
    this.reconnectingHandlers = [];
    this.reconnectedHandlers = [];
    this.reconnectFailedHandlers = [];
    this._isConnecting = false;
    this._isConnected = false;
    this._explicitClose = false;
    this._reconnectConfig = { ...DEFAULT_RECONNECT, ...reconnect };
    this._reconnectTimer = null;
    this._reconnectAttempt = 0;
  }

  /**
   * Establish WebSocket connection
   *
   * If already connecting or connected, this is a no-op.
   * All handlers should be registered before calling connect().
   */
  connect(): void {
    if (this._isConnecting || this._isConnected) {
      return;
    }

    this._isConnecting = true;

    const ws = new WebSocket(this.url);
    this.ws = ws;

    // Set binary type to arraybuffer for v0 protocol binary image handling
    ws.binaryType = 'arraybuffer';

    ws.onopen = (event: Event) => {
      this._isConnecting = false;
      this._isConnected = true;

      const wasReconnect = this._reconnectAttempt > 0;
      this._reconnectAttempt = 0;

      if (wasReconnect) {
        // Notify reconnected handlers *before* the generic open handlers
        // so the Figure can prepare state before the init handshake fires.
        this.reconnectedHandlers.forEach((handler) => handler());
      }

      this.openHandlers.forEach((handler) => {
        handler(event);
      });
    };

    ws.onmessage = (event: MessageEvent) => {
      this.messageHandlers.forEach((handler) => {
        handler(event);
      });
    };

    ws.onclose = (event: CloseEvent) => {
      const wasConnected = this._isConnected;
      this._isConnected = false;
      this._isConnecting = false;

      this.closeHandlers.forEach((handler) => {
        handler(event);
      });

      // Start / continue reconnection if the close was unexpected.
      // Two cases:
      //   1. wasConnected — a live connection dropped (first disconnect)
      //   2. _reconnectAttempt > 0 — a reconnect attempt failed to open
      //      (onerror → onclose with wasConnected=false)
      if (
        !this._explicitClose &&
        (wasConnected || this._reconnectAttempt > 0) &&
        this._reconnectConfig.maxAttempts > 0
      ) {
        this._startReconnect();
      }
    };

    ws.onerror = (event: Event) => {
      // If we're in the middle of a reconnect attempt, the error will be
      // followed by an onclose — which will drive the next attempt.  We
      // still fire the error handlers for logging purposes.
      this.errorHandlers.forEach((handler) => {
        handler(event);
      });
    };
  }

  /**
   * Register handler for connection open event
   */
  onOpen(handler: (event: Event) => void): void {
    this.openHandlers.push(handler);
  }

  /**
   * Register handler for incoming messages
   */
  onMessage(handler: (event: MessageEvent) => void): void {
    this.messageHandlers.push(handler);
  }

  /**
   * Register handler for connection close event
   */
  onClose(handler: (event: CloseEvent) => void): void {
    this.closeHandlers.push(handler);
  }

  /**
   * Register handler for connection error event
   */
  onError(handler: (event: Event) => void): void {
    this.errorHandlers.push(handler);
  }

  /**
   * Register handler called at the start of each reconnection attempt.
   *
   * @param handler - ``(attempt, maxAttempts) => void``
   */
  onReconnecting(handler: (attempt: number, maxAttempts: number) => void): void {
    this.reconnectingHandlers.push(handler);
  }

  /**
   * Register handler called when reconnection succeeds.
   *
   * Fired *before* the normal ``onOpen`` handlers so the Figure can
   * prepare for a fresh handshake.
   */
  onReconnected(handler: () => void): void {
    this.reconnectedHandlers.push(handler);
  }

  /**
   * Register handler called when all reconnection attempts are exhausted.
   */
  onReconnectFailed(handler: () => void): void {
    this.reconnectFailedHandlers.push(handler);
  }

  /**
   * Send data through WebSocket
   *
   * @param data - Data to send (will be passed directly to WebSocket.send())
   */
  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    if (this.ws && this._isConnected) {
      this.ws.send(data);
    }
  }

  /**
   * Close WebSocket connection (suppresses reconnection)
   */
  close(): void {
    this._explicitClose = true;
    this._cancelReconnect();
    if (this.ws) {
      this.ws.close();
    }
  }

  /**
   * Get current WebSocket ready state
   *
   * @returns WebSocket.CONNECTING, OPEN, CLOSING, or CLOSED
   */
  getReadyState(): number {
    return this.ws ? this.ws.readyState : WebSocket.CLOSED;
  }

  /**
   * Check if WebSocket is currently connected
   */
  isConnected(): boolean {
    return this._isConnected;
  }

  /**
   * Whether a reconnection cycle is active.
   *
   * Returns ``true`` from the first ``onclose`` that triggers reconnection
   * until reconnection succeeds, is cancelled, or all attempts are
   * exhausted.  This is safe to query from ``onClose`` / ``onError``
   * handlers to decide whether to tear down resources.
   */
  isReconnecting(): boolean {
    return this._reconnectAttempt > 0 || this._reconnectTimer !== null;
  }

  // -- reconnection internals -----------------------------------------------

  /**
   * Schedule the next reconnection attempt using exponential backoff with
   * jitter.  Increments the attempt counter and, if the limit is reached,
   * fires ``onReconnectFailed`` handlers instead.
   */
  private _startReconnect(): void {
    const { maxAttempts } = this._reconnectConfig;

    this._reconnectAttempt += 1;

    if (this._reconnectAttempt > maxAttempts) {
      // All attempts exhausted
      console.warn(`WebSocketManager: reconnect failed after ${maxAttempts} attempts`);
      this._reconnectAttempt = 0;
      this.reconnectFailedHandlers.forEach((handler) => handler());
      return;
    }

    // Notify listeners of the upcoming attempt
    this.reconnectingHandlers.forEach((handler) =>
      handler(this._reconnectAttempt, maxAttempts)
    );

    const delay = this._computeDelay(this._reconnectAttempt);
    console.log(
      `WebSocketManager: reconnect attempt ${this._reconnectAttempt}/${maxAttempts} in ${Math.round(delay)} ms`
    );

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      // connect() checks _isConnecting / _isConnected — both are false here
      this.connect();
    }, delay);
  }

  /**
   * Compute the backoff delay (ms) for the given attempt number.
   *
   * Uses exponential backoff capped at ``ReconnectConfig.maxDelayMs`` with
   * ±25% random jitter.
   *
   * @param attempt - 1-based reconnection attempt number.
   * @returns Delay in milliseconds.
   */
  private _computeDelay(attempt: number): number {
    const { initialDelayMs, maxDelayMs, backoffBase } = this._reconnectConfig;
    let delay = Math.min(
      initialDelayMs * Math.pow(backoffBase, attempt - 1),
      maxDelayMs
    );
    // ±25% jitter
    delay *= 1.0 + 0.25 * (2.0 * Math.random() - 1.0);
    return delay;
  }

  /**
   * Cancel any pending reconnection timer and reset the attempt counter.
   * Called by ``close()`` to prevent reconnection after an intentional disconnect.
   */
  private _cancelReconnect(): void {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this._reconnectAttempt = 0;
  }

  /**
   * Get the underlying WebSocket instance (if any).
   *
   * This is primarily useful for inspecting readyState or binaryType.
   * Prefer using the manager's own ``send`` / ``close`` methods.
   */
  get rawSocket(): WebSocket | null {
    return this.ws;
  }
}
