/**
 * MatplotlibEmbeddable - Unified matplotlib component
 *
 * This is a STUB implementation that will be completed in Phase 4.
 * For now, it provides just enough structure to make the build succeed.
 */

import type { EmbeddableConfig } from './types.js';

/**
 * Main embeddable matplotlib component class
 *
 * This provides a unified API for embedding interactive matplotlib figures
 * into any web application, supporting both template-based and programmatic usage.
 */
export class MatplotlibEmbeddable {
  private readonly config: Required<EmbeddableConfig>;

  constructor(config: EmbeddableConfig) {
    // Set defaults for optional properties
    this.config = {
      container: config.container,
      plotName: config.plotName,
      baseUrl: config.baseUrl ?? '',
      initParams: config.initParams ?? {},
      updateParams: config.updateParams ?? {},
      onConnect: config.onConnect ?? (() => {}),
      onError: config.onError ?? ((err: Error) => console.error('MPL Error:', err)),
      onUpdate: config.onUpdate ?? (() => {}),
      onDisconnect: config.onDisconnect ?? (() => {}),
      showToolbar: config.showToolbar ?? true,
      showUpdateForm: config.showUpdateForm ?? true,
      staticPath: config.staticPath ?? '/mpl-static',
      autoConnect: config.autoConnect ?? true,
    };

    // Validate required options
    if (!this.config.container) {
      throw new Error('container option is required');
    }
    if (!this.config.plotName) {
      throw new Error('plotName option is required');
    }

    // TODO: Phase 3-4 - Implement full functionality
    // This is a minimal stub for build system verification only
    // Actual implementation will be added when converting mpl.js components
  }

  /**
   * Connect to the WebSocket and initialize the plot
   */
  async connect(): Promise<void> {
    // Stub - no-op
  }

  /**
   * Update plot parameters
   */
  update(params: Record<string, unknown>): void {
    // Stub - no-op
    void params;
  }

  /**
   * Disconnect and cleanup
   */
  disconnect(): void {
    // Stub - no-op
  }

  /**
   * Check if currently connected
   */
  isConnected(): boolean {
    return false;
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
