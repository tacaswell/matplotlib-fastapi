/**
 * matplotlib-fastapi TypeScript Components
 *
 * Main entry point for the unified matplotlib embeddable component.
 * This file will be compiled to component.js and exposed globally.
 */

// Export core components
export { WebSocketManager, type ReconnectConfig } from './websocket-manager.js';
export { Figure } from './figure.js';
export { MatplotlibClient } from './client.js';
export { listPlots } from './discovery.js';
export { VERSION } from './_version.js';

// Export types for TypeScript users
export type {
  ClientConfig,
  PlotSchemaResponse,
  PlotListEntry,
  PlotsListResponse,
  JSONSchema,
  JSONSchemaProperty,
  ImageMode,
  NavigationMode,
  ServerMessage,
  ClientMessage,
} from './types.js';

// Expose to global window for script tag usage
import { WebSocketManager } from './websocket-manager.js';
import { Figure } from './figure.js';
import { MatplotlibClient } from './client.js';
import { listPlots } from './discovery.js';
import { VERSION } from './_version.js';

declare global {
  interface Window {
    MatplotlibClient: typeof MatplotlibClient;
    mpl: {
      WebSocketManager: typeof WebSocketManager;
      Figure: typeof Figure;
      listPlots: typeof listPlots;
      VERSION: string;
    };
  }
}

if (typeof window !== 'undefined') {
  window.MatplotlibClient = MatplotlibClient;

  // Initialize mpl namespace if not already present
  // Note: Toolbar configuration is now sent via WebSocket messages
  if (!window.mpl) {
    window.mpl = {} as any;
  }

  // Preserve any properties already set by Python (toolbar config)
  window.mpl.WebSocketManager = WebSocketManager;
  window.mpl.Figure = Figure;
  window.mpl.listPlots = listPlots;
  window.mpl.VERSION = VERSION;
}
