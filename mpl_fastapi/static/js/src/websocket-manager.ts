/**
 * WebSocketManager - Manages WebSocket connection lifecycle
 *
 * This class ensures proper ordering of connection and message handling by
 * allowing handlers to be registered before the connection is established.
 * This prevents race conditions where messages might arrive before handlers
 * are ready.
 */
export class WebSocketManager {
  private url: string;
  private ws: WebSocket | null;
  private messageHandlers: Array<(event: MessageEvent) => void>;
  private openHandlers: Array<(event: Event) => void>;
  private closeHandlers: Array<(event: CloseEvent) => void>;
  private errorHandlers: Array<(event: Event) => void>;
  private _isConnecting: boolean;
  private _isConnected: boolean;

  constructor(url: string) {
    this.url = url;
    this.ws = null;
    this.messageHandlers = [];
    this.openHandlers = [];
    this.closeHandlers = [];
    this.errorHandlers = [];
    this._isConnecting = false;
    this._isConnected = false;
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

    const WebSocketType = this.getWebSocketType();
    this.ws = new WebSocketType(this.url);

    this.ws.onopen = (event: Event) => {
      this._isConnecting = false;
      this._isConnected = true;
      this.openHandlers.forEach((handler) => {
        handler(event);
      });
    };

    this.ws.onmessage = (event: MessageEvent) => {
      this.messageHandlers.forEach((handler) => {
        handler(event);
      });
    };

    this.ws.onclose = (event: CloseEvent) => {
      this._isConnected = false;
      this.closeHandlers.forEach((handler) => {
        handler(event);
      });
    };

    this.ws.onerror = (event: Event) => {
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
   * Close WebSocket connection
   */
  close(): void {
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
   * Get appropriate WebSocket constructor for the browser
   *
   * @returns WebSocket constructor
   * @throws Error if WebSocket is not supported
   */
  private getWebSocketType(): typeof WebSocket {
    if (typeof WebSocket !== 'undefined') {
      return WebSocket;
    } else if (typeof (window as any).MozWebSocket !== 'undefined') {
      return (window as any).MozWebSocket;
    } else {
      throw new Error(
        'Your browser does not have WebSocket support. ' +
          'Please try Chrome, Safari or Firefox ≥ 6.'
      );
    }
  }
}
