/**
 * matplotlib-fastapi TypeScript Components
 *
 * Main entry point for the unified matplotlib embeddable component.
 * This file will be compiled to component.js and exposed globally.
 */

// Export core components
export { WebSocketManager } from './websocket-manager.js';
export { Figure } from './figure.js';
export { MatplotlibEmbeddable } from './embeddable.js';

// Export types for TypeScript users
export type {
  EmbeddableConfig,
  PlotSchemaResponse,
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
import { MatplotlibEmbeddable } from './embeddable.js';

declare global {
  interface Window {
    MatplotlibEmbeddable: typeof MatplotlibEmbeddable;
    mpl: {
      WebSocketManager: typeof WebSocketManager;
      Figure: typeof Figure;
    };
  }
}

if (typeof window !== 'undefined') {
  window.MatplotlibEmbeddable = MatplotlibEmbeddable;

  // Initialize mpl namespace if not already present
  // Note: Toolbar configuration is now sent via WebSocket messages
  if (!window.mpl) {
    window.mpl = {} as any;
  }

  // Preserve any properties already set by Python (toolbar config)
  window.mpl.WebSocketManager = WebSocketManager;
  window.mpl.Figure = Figure;
}
