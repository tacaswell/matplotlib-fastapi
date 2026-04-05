/**
 * Plot discovery — query a mpl_fastapi server for available plots.
 *
 * @example
 * ```typescript
 * import { listPlots } from 'mpl-fastapi';
 *
 * const response = await listPlots('/plots');
 * for (const [name, info] of Object.entries(response.plots)) {
 *   console.log(`${name}: ${info.description}`);
 *   console.log(`  WS: ${info.ws_url}`);
 *   console.log(`  Init schema: ${JSON.stringify(info.parameters)}`);
 * }
 * ```
 */

import type { PlotsListResponse } from './types.js';

/**
 * Fetch the list of available plots from a mpl_fastapi server.
 *
 * Calls the ``GET /plots`` JSON endpoint and returns the typed
 * response containing plot names, descriptions, schemas, and URLs.
 *
 * @param baseUrl - Base URL where the mpl router is mounted (e.g. ``'/plots'``
 *   or ``'http://localhost:8000/plots'``).
 * @param token - Optional authentication token (sent as ``Bearer`` header).
 * @returns Parsed ``PlotsListResponse``.
 * @throws {Error} If the fetch fails or the server returns a non-OK status.
 */
export async function listPlots(
  baseUrl: string,
  token?: string,
): Promise<PlotsListResponse> {
  const headers: Record<string, string> = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const url = `${baseUrl.replace(/\/+$/, '')}/plots`;
  const response = await fetch(url, { headers });

  if (!response.ok) {
    throw new Error(
      `Failed to list plots: ${response.status} ${response.statusText}`,
    );
  }

  return (await response.json()) as PlotsListResponse;
}
