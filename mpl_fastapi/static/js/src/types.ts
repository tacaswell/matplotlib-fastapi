/**
 * Type definitions for matplotlib-fastapi
 *
 * This file contains all shared TypeScript interfaces, types, and enums
 * used across the matplotlib-fastapi JavaScript components.
 */

import type { ReconnectConfig } from './websocket-manager.js';

// ============================================================================
// Protocol Constants (v0)
// ============================================================================

/** Current protocol version */
export const PROTOCOL_VERSION = 0;

/** Binary image type/mode - first byte of header */
export enum ImageTypeMode {
  FULL = 0x00,
  DIFF = 0x01,
}

/** Binary image format - second byte of header */
export enum ImageFormat {
  PNG = 0x01,
  JPEG = 0x02,
  WEBP = 0x03,
}

/**
 * Parsed binary image header (8 bytes total)
 * Structure: [type_mode: 1, format: 1, seq_num: 2, base_seq: 2, flags: 2]
 */
export interface BinaryImageHeader {
  /** Image type: FULL (0x00) or DIFF (0x01) */
  typeMode: ImageTypeMode;
  /** Image format: PNG (0x01), JPEG (0x02), WEBP (0x03) */
  format: ImageFormat;
  /** Sequence number (1-65535, 0 reserved) */
  seqNum: number;
  /** Base sequence for diffs (0 for FULL) */
  baseSeq: number;
  /** Reserved flags */
  flags: number;
}

/** Binary header size in bytes */
export const BINARY_HEADER_SIZE = 8;

/**
 * Parse binary image header from ArrayBuffer
 * @param buffer - ArrayBuffer containing at least 8 bytes
 * @returns Parsed header and image data
 */
export function parseBinaryImage(buffer: ArrayBuffer): {
  header: BinaryImageHeader;
  imageData: Uint8Array;
} {
  const view = new DataView(buffer);
  const header: BinaryImageHeader = {
    typeMode: view.getUint8(0) as ImageTypeMode,
    format: view.getUint8(1) as ImageFormat,
    seqNum: view.getUint16(2, false), // big-endian
    baseSeq: view.getUint16(4, false), // big-endian
    flags: view.getUint16(6, false), // big-endian
  };
  const imageData = new Uint8Array(buffer, BINARY_HEADER_SIZE);
  return { header, imageData };
}

/**
 * Get MIME type string for image format
 */
export function getImageMimeType(format: ImageFormat): string {
  switch (format) {
    case ImageFormat.PNG:
      return 'image/png';
    case ImageFormat.JPEG:
      return 'image/jpeg';
    case ImageFormat.WEBP:
      return 'image/webp';
    default:
      return 'image/png';
  }
}

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * Configuration options for MatplotlibEmbeddable component
 */
export interface EmbeddableConfig {
  /** The DOM element to render the plot into */
  container: HTMLElement;

  /** Name of the plot to load (must match server-side configuration) */
  plotName: string;

  /** Base URL for API endpoints (default: '') */
  baseUrl?: string;

  /** Initial parameters for plot generation */
  initParams?: Record<string, unknown>;

  /** Initial update parameters */
  updateParams?: Record<string, unknown>;

  /** Callback when WebSocket connection is established */
  onConnect?: () => void;

  /** Callback when an error occurs */
  onError?: (error: Error) => void;

  /** Callback when plot parameters are updated */
  onUpdate?: (params: Record<string, unknown>) => void;

  /** Callback when WebSocket connection is closed */
  onDisconnect?: () => void;

  /** Whether to show the matplotlib toolbar (default: true) */
  showToolbar?: boolean;

  /** Whether to show auto-generated update form (default: true) */
  showUpdateForm?: boolean;

  /** Path to static assets (default: '/mpl-static') */
  staticPath?: string;

  /** Whether to automatically connect on instantiation (default: true) */
  autoConnect?: boolean;

  /** Reconnection configuration (default: enabled with 10 attempts) */
  reconnect?: ReconnectConfig;

  /** Callback when reconnection attempt starts */
  onReconnecting?: (attempt: number, maxAttempts: number) => void;

  /** Callback when reconnection succeeds */
  onReconnected?: () => void;

  /** Callback when all reconnection attempts are exhausted */
  onReconnectFailed?: () => void;
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

export type ImageMode = 'full' | 'diff';
export type NavigationMode = 'PAN' | 'ZOOM' | null;

/**
 * Base interface for all WebSocket messages
 */
export interface BaseMessage {
  type: string;
}

// ============================================================================
// Protocol v0 Consolidated Messages
// ============================================================================

/**
 * Client → Server: Initial handshake message (MUST be first message)
 */
export interface InitMessage extends BaseMessage {
  type: 'init';
  /** Protocol version (must match server) */
  protocol_version: number;
  /** Device pixel ratio for HiDPI displays (default: 1.0) */
  device_pixel_ratio?: number;
  /** Whether client supports binary WebSocket messages (default: true) */
  supports_binary?: boolean;
}

/**
 * Server → Client: Consolidated configuration response
 */
export interface ConfigMessage extends BaseMessage {
  type: 'config';
  /** Protocol version */
  protocol_version: number;
  /** Unique connection ID for this session */
  connection_id: string;
  /** Figure configuration */
  figure: {
    /** Figure size in CSS pixels [width, height] */
    size: [number, number];
    /** Figure DPI */
    dpi: number;
    /** Figure label/title */
    label: string;
  };
  /** Toolbar configuration */
  toolbar: {
    /** Array of toolbar item definitions [name, tooltip, icon, method] */
    items: Array<[string, string, string, string]>;
    /** Initial navigation history state */
    history: {
      back: boolean;
      forward: boolean;
    };
  };
  /** Save/download configuration */
  save: {
    /** Supported file formats */
    formats: string[];
    /** Default format */
    default_format: string;
  };
  /** Image encoding configuration */
  image: {
    /** Image format (png, jpeg, webp) */
    format: string;
  };
  /** JSON schema for update parameters (null if not supported) */
  update_schema: Record<string, unknown> | null;
}

/**
 * Server → Client: Error message
 */
export interface ErrorMessage extends BaseMessage {
  type: 'error';
  message: string;
}

/**
 * Server → Client: Image mode notification
 */
export interface ImageModeMessage extends BaseMessage {
  type: 'image_mode';
  mode: ImageMode;
}

/**
 * Server → Client: Connection ID for download endpoint
 */
export interface ConnectionIdMessage extends BaseMessage {
  type: 'connection_id';
  id: string;
}

/**
 * Server → Client: Update figure label/title
 */
export interface FigureLabelMessage extends BaseMessage {
  type: 'figure_label';
  label: string;
}

/**
 * Server → Client: Update cursor style
 */
export interface CursorMessage extends BaseMessage {
  type: 'cursor';
  cursor: string;
}

/**
 * Server → Client: Display message (e.g., coordinates)
 */
export interface StatusMessage extends BaseMessage {
  type: 'message';
  message: string;
}

/**
 * Server → Client: Draw rubberband rectangle
 */
export interface RubberbandMessage extends BaseMessage {
  type: 'rubberband';
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Server → Client: Update history button states
 */
export interface HistoryButtonsMessage extends BaseMessage {
  type: 'history_buttons';
  [key: string]: boolean | string;
}

/**
 * Server → Client: Update navigation mode
 */
export interface NavigateModeMessage extends BaseMessage {
  type: 'navigate_mode';
  mode: NavigationMode;
}

/**
 * Server → Client: Notify that figure has changed (debounce-able)
 */
export interface InvalidateMessage extends BaseMessage {
  type: 'invalidate';
}

/**
 * Server → Client: Resize notification
 */
export interface ResizeMessage extends BaseMessage {
  type: 'resize';
  width: number;
  height: number;
}

/**
 * Server → Client: Toolbar configuration
 */
export interface ToolbarConfigMessage extends BaseMessage {
  type: 'toolbar_config';
  items: Array<[string, string, string, string]>;
}

/**
 * Server → Client: Supported save file formats
 */
export interface SaveFormatsMessage extends BaseMessage {
  type: 'save_formats';
  formats: string[];
}

/**
 * Server → Client: Default save file format
 */
export interface DefaultSaveFormatMessage extends BaseMessage {
  type: 'default_save_format';
  format: string;
}

/**
 * Server → Client: Save operation completed successfully
 */
export interface SaveCompleteMessage extends BaseMessage {
  type: 'save_complete';
  file_id: string;
  download_url: string;
  filename: string;
  format: string;
}

/**
 * Server → Client: Save operation failed
 */
export interface SaveErrorMessage extends BaseMessage {
  type: 'save_error';
  message: string;
}

/**
 * Union type of all server messages
 */
export type ServerMessage =
  | ConfigMessage
  | ErrorMessage
  | ImageModeMessage
  | ConnectionIdMessage
  | FigureLabelMessage
  | CursorMessage
  | StatusMessage
  | RubberbandMessage
  | HistoryButtonsMessage
  | NavigateModeMessage
  | InvalidateMessage
  | ResizeMessage
  | ToolbarConfigMessage
  | SaveFormatsMessage
  | DefaultSaveFormatMessage
  | SaveCompleteMessage
  | SaveErrorMessage;

/**
 * Client → Server: Binary support notification
 */
export interface SupportsBinaryMessage extends BaseMessage {
  type: 'supports_binary';
  value: boolean;
}

/**
 * Client → Server: Request image mode
 */
export interface SendImageModeMessage extends BaseMessage {
  type: 'send_image_mode';
}

/**
 * Client → Server: Set device pixel ratio
 */
export interface SetDevicePixelRatioMessage extends BaseMessage {
  type: 'set_device_pixel_ratio';
  device_pixel_ratio: number;
}

/**
 * Client → Server: Request refresh/initial render with metadata
 */
export interface RefreshMessage extends BaseMessage {
  type: 'refresh';
}

/**
 * Client → Server: Request immediate render (returns PNG)
 */
export interface RenderMessage extends BaseMessage {
  type: 'render';
}

/**
 * Client → Server: Mouse button press
 */
export interface ButtonPressMessage extends BaseMessage {
  type: 'button_press';
  x: number;
  y: number;
  button: number;
  guiEvent?: Record<string, unknown>;
}

/**
 * Client → Server: Mouse motion
 */
export interface MotionNotifyMessage extends BaseMessage {
  type: 'motion_notify';
  x: number;
  y: number;
  guiEvent?: Record<string, unknown>;
}

/**
 * Client → Server: Mouse button release
 */
export interface ButtonReleaseMessage extends BaseMessage {
  type: 'button_release';
  x: number;
  y: number;
  button: number;
  guiEvent?: Record<string, unknown>;
}

/**
 * Client → Server: Scroll/wheel event
 */
export interface ScrollMessage extends BaseMessage {
  type: 'scroll';
  x: number;
  y: number;
  step: number;
  guiEvent?: Record<string, unknown>;
}

/**
 * Client → Server: Key press
 */
export interface KeyPressMessage extends BaseMessage {
  type: 'key_press';
  key: string;
  guiEvent?: Record<string, unknown>;
}

/**
 * Client → Server: Key release
 */
export interface KeyReleaseMessage extends BaseMessage {
  type: 'key_release';
  key: string;
  guiEvent?: Record<string, unknown>;
}

/**
 * Client → Server: Toolbar button click
 */
export interface ToolbarButtonMessage extends BaseMessage {
  type: 'toolbar_button';
  name: string;
}

/**
 * Client → Server: Update plot parameters
 */
export interface UpdateParamsMessage extends BaseMessage {
  type: 'update_params';
  params: Record<string, unknown>;
}

/**
 * Client → Server: Save figure to file
 */
export interface SaveFigureMessage extends BaseMessage {
  type: 'save_figure';
  format: string;
  dpi?: number;
  transparent?: boolean;
}

/**
 * Client → Server: Acknowledge message
 */
export interface AckMessage extends BaseMessage {
  type: 'ack';
}

/**
 * Union type of all client messages
 */
export type ClientMessage =
  | InitMessage
  | SupportsBinaryMessage
  | SendImageModeMessage
  | SetDevicePixelRatioMessage
  | RefreshMessage
  | RenderMessage
  | ButtonPressMessage
  | MotionNotifyMessage
  | ButtonReleaseMessage
  | ScrollMessage
  | KeyPressMessage
  | KeyReleaseMessage
  | ToolbarButtonMessage
  | UpdateParamsMessage
  | SaveFigureMessage
  | AckMessage;

// ============================================================================
// JSON Schema Types (for form generation)
// ============================================================================

/**
 * JSON Schema property definition
 */
export interface JSONSchemaProperty {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  enum?: unknown[];
  items?: JSONSchemaProperty;
  properties?: Record<string, JSONSchemaProperty>;
  required?: string[];
}

/**
 * JSON Schema definition
 */
export interface JSONSchema {
  type?: string;
  title?: string;
  description?: string;
  properties?: Record<string, JSONSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean;
}

/**
 * Response from /api/plots/{name}/schema endpoint
 */
export interface PlotSchemaResponse {
  plot_name: string;
  description: string;
  init_schema: JSONSchema;
  update_schema: JSONSchema | null;
}

// ============================================================================
// Internal Component State Types
// ============================================================================

/**
 * Internal state for MatplotlibEmbeddable component
 */
export interface ComponentState {
  connected: boolean;
  connectionId: string | null;
}

/**
 * Internal references to DOM elements and objects
 */
export interface ComponentRefs {
  ws_manager: WebSocketManager | null;
  figure: Figure | null;
  submitButton: HTMLButtonElement | null;
}

// ============================================================================
// Forward declarations for classes (defined in other files)
// ============================================================================

/**
 * WebSocket lifecycle manager
 */
export interface WebSocketManager {
  connect(): void;
  close(): void;
  send(data: string): void;
  isConnected(): boolean;
  isReconnecting(): boolean;
  onOpen(handler: (event: Event) => void): void;
  onMessage(handler: (event: MessageEvent) => void): void;
  onClose(handler: (event: CloseEvent) => void): void;
  onError(handler: (event: Event) => void): void;
  onReconnecting(handler: (attempt: number, maxAttempts: number) => void): void;
  onReconnected(handler: () => void): void;
  onReconnectFailed(handler: () => void): void;
}

/**
 * Matplotlib figure renderer
 */
export interface Figure {
  id: string;
  connection_id: string | null;
  send_message(type: string, properties: Record<string, unknown>): void;
  canvas_div: HTMLDivElement | undefined;
  root: HTMLDivElement;
}
