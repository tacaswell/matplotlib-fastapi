/**
 * matplotlib-fastapi TypeScript Components
 *
 * Main entry point for the unified matplotlib embeddable component.
 * This file will be compiled to component.js and exposed globally.
 */

// Export MatplotlibEmbeddable (will be implemented in Phase 4)
export { MatplotlibEmbeddable } from './embeddable.js';

// Export types for TypeScript users
export type {
  EmbeddableConfig,
  PlotSchemaResponse,
  JSONSchema,
  JSONSchemaProperty,
} from './types.js';

// Expose MatplotlibEmbeddable globally for script tag usage
import { MatplotlibEmbeddable } from './embeddable.js';

declare global {
  interface Window {
    MatplotlibEmbeddable: typeof MatplotlibEmbeddable;
  }
}

if (typeof window !== 'undefined') {
  window.MatplotlibEmbeddable = MatplotlibEmbeddable;
}
