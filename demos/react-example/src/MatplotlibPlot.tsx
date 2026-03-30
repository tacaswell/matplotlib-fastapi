import React, { useEffect, useRef, useCallback } from 'react';
import { MatplotlibEmbeddable, EmbeddableConfig } from 'mpl-fastapi';

export interface MatplotlibPlotProps {
  /** Name of the plot registered with create_mpl_router */
  plotName: string;
  /** Base URL where the mpl router is mounted (e.g., '/plots') */
  baseUrl?: string;
  /** Authentication token for protected routers */
  token?: string;
  /** Initial parameters passed when creating the figure */
  initParams?: Record<string, unknown>;
  /** Update parameters for dynamic updates (without reconnecting) */
  updateParams?: Record<string, unknown>;
  /** Show the matplotlib navigation toolbar */
  showToolbar?: boolean;
  /** Show the auto-generated update form */
  showUpdateForm?: boolean;
  /** Path to matplotlib static files */
  staticPath?: string;
  /** Callback when WebSocket connects */
  onConnect?: () => void;
  /** Callback on error */
  onError?: (error: Error) => void;
  /** Callback when parameters are updated */
  onUpdate?: (params: Record<string, unknown>) => void;
  /** Callback when WebSocket disconnects */
  onDisconnect?: () => void;
  /** Container style */
  style?: React.CSSProperties;
  /** Container className */
  className?: string;
}

/**
 * React wrapper component for MatplotlibEmbeddable.
 * 
 * This component manages the lifecycle of the embeddable plot,
 * connecting on mount and disconnecting on unmount.
 * 
 * - `initParams` are sent at connection time and require reconnection to change
 * - `updateParams` can be changed at any time without reconnecting
 * 
 * @example
 * ```tsx
 * <MatplotlibPlot
 *   plotName="sine"
 *   baseUrl="/plots"
 *   initParams={{ frequency: 2.0 }}
 *   updateParams={{ phase }}
 *   onConnect={() => console.log('Connected!')}
 * />
 * ```
 */
export function MatplotlibPlot({
  plotName,
  baseUrl = '',
  token,
  initParams = {},
  updateParams = {},
  showToolbar = true,
  showUpdateForm = false,
  staticPath = '/mpl-static',
  onConnect,
  onError,
  onUpdate,
  onDisconnect,
  style,
  className,
}: MatplotlibPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<MatplotlibEmbeddable | null>(null);
  const initializedRef = useRef(false);

  // Memoize callbacks to avoid unnecessary reconnections
  const handleConnect = useCallback(() => onConnect?.(), [onConnect]);
  const handleError = useCallback((error: Error) => onError?.(error), [onError]);
  const handleUpdate = useCallback((params: Record<string, unknown>) => onUpdate?.(params), [onUpdate]);
  const handleDisconnect = useCallback(() => onDisconnect?.(), [onDisconnect]);

  // Initialize the plot on mount (only once)
  useEffect(() => {
    if (!containerRef.current || initializedRef.current) return;
    initializedRef.current = true;

    const config: EmbeddableConfig = {
      container: containerRef.current,
      plotName,
      baseUrl,
      token,
      initParams,
      updateParams,
      showToolbar,
      showUpdateForm,
      staticPath,
      onConnect: handleConnect,
      onError: handleError,
      onUpdate: handleUpdate,
      onDisconnect: handleDisconnect,
      autoConnect: true,
    };

    plotRef.current = new MatplotlibEmbeddable(config);

    // Cleanup on unmount only
    return () => {
      if (plotRef.current) {
        plotRef.current.disconnect();
        plotRef.current = null;
      }
      initializedRef.current = false;
    };
  }, [plotName, baseUrl]); // Only reconnect if plot name or base URL changes

  // Update parameters when updateParams prop changes (without reconnecting)
  useEffect(() => {
    if (plotRef.current?.isConnected() && Object.keys(updateParams).length > 0) {
      plotRef.current.update(updateParams);
    }
  }, [updateParams]);

  return (
    <div 
      ref={containerRef} 
      style={style} 
      className={className}
    />
  );
}
