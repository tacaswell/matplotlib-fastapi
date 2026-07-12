import React, { useEffect, useState, useMemo, useCallback } from 'react';
import {
  listPlots,
  type PlotsListResponse,
  type PlotListEntry,
  type JSONSchemaProperty,
} from 'mpl-fastapi';
import { MatplotlibPlot } from './MatplotlibPlot';

interface OpenedPlot {
  id: string;
  name: string;
  initParams: Record<string, unknown>;
  updateParams: Record<string, unknown>;
}

interface PlotPickerProps {
  /** Base URL where the mpl router is mounted (e.g. '/plots') */
  baseUrl?: string;
  /** Authentication token */
  token?: string;
}

/**
 * Schema-driven parameter form — renders inputs for each property
 * in a JSON Schema.
 */
function SchemaForm({
  schema,
  values,
  onChange,
}: {
  schema: Record<string, JSONSchemaProperty>;
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
}) {
  return (
    <>
      {Object.entries(schema).map(([name, prop]) => {
        const label = prop.title || name;
        const desc = prop.description;

        if (prop.type === 'boolean') {
          return (
            <div key={name} style={{ marginBottom: 8 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input
                  type="checkbox"
                  checked={Boolean(values[name] ?? prop.default ?? false)}
                  onChange={(e) => onChange(name, e.target.checked)}
                />
                <span>
                  {label}
                  {desc && (
                    <span style={{ color: '#888', fontSize: '0.9em' }}> ({desc})</span>
                  )}
                </span>
              </label>
            </div>
          );
        }

        if (prop.enum) {
          return (
            <div key={name} style={{ marginBottom: 8 }}>
              <label style={{ display: 'block', marginBottom: 4 }}>
                {label}
                {desc && (
                  <span style={{ color: '#888', fontSize: '0.9em' }}> ({desc})</span>
                )}
              </label>
              <select
                value={String(values[name] ?? prop.default ?? '')}
                onChange={(e) => onChange(name, e.target.value)}
                style={{ padding: '4px 8px', width: '100%' }}
              >
                {prop.enum.map((v) => (
                  <option key={String(v)} value={String(v)}>
                    {String(v)}
                  </option>
                ))}
              </select>
            </div>
          );
        }

        if (prop.type === 'number' || prop.type === 'integer') {
          return (
            <div key={name} style={{ marginBottom: 8 }}>
              <label style={{ display: 'block', marginBottom: 4 }}>
                {label}
                {desc && (
                  <span style={{ color: '#888', fontSize: '0.9em' }}> ({desc})</span>
                )}
              </label>
              <input
                type="number"
                step={prop.type === 'integer' ? '1' : 'any'}
                min={prop.minimum}
                max={prop.maximum}
                value={String(values[name] ?? prop.default ?? '')}
                onChange={(e) => {
                  const v =
                    prop.type === 'integer'
                      ? parseInt(e.target.value, 10)
                      : parseFloat(e.target.value);
                  if (!isNaN(v)) onChange(name, v);
                }}
                style={{ padding: '4px 8px', width: '100%' }}
              />
            </div>
          );
        }

        // Default: string input
        return (
          <div key={name} style={{ marginBottom: 8 }}>
            <label style={{ display: 'block', marginBottom: 4 }}>
              {label}
              {desc && (
                <span style={{ color: '#888', fontSize: '0.9em' }}> ({desc})</span>
              )}
            </label>
            <input
              type="text"
              value={String(values[name] ?? prop.default ?? '')}
              onChange={(e) => onChange(name, e.target.value)}
              style={{ padding: '4px 8px', width: '100%' }}
            />
          </div>
        );
      })}
    </>
  );
}

/**
 * PlotPicker — discover available plots on a server and open them.
 *
 * Calls `listPlots()` on mount, renders a selectable list of plots,
 * and provides auto-generated parameter forms from the schemas.
 * Multiple plots can be opened simultaneously.
 */
export function PlotPicker({ baseUrl = '/plots', token }: PlotPickerProps) {
  const [plots, setPlots] = useState<Record<string, PlotListEntry>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [initValues, setInitValues] = useState<Record<string, unknown>>({});
  const [updateValues, setUpdateValues] = useState<Record<string, unknown>>({});
  const [openedPlots, setOpenedPlots] = useState<OpenedPlot[]>([]);

  // Fetch plot list on mount
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listPlots(baseUrl, token)
      .then((resp) => {
        if (!cancelled) {
          setPlots(resp.plots);
          const names = Object.keys(resp.plots);
          if (names.length > 0) {
            setSelectedName(names[0]);
            _resetFormValues(resp.plots[names[0]]);
          }
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(String(err));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, token]);

  const _resetFormValues = useCallback((entry: PlotListEntry) => {
    const initDefaults: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(entry.parameters?.properties ?? {})) {
      if (v.default !== undefined) initDefaults[k] = v.default;
    }
    setInitValues(initDefaults);

    const updateDefaults: Record<string, unknown> = {};
    if (entry.update_schema) {
      for (const [k, v] of Object.entries(entry.update_schema.properties ?? {})) {
        if (v.default !== undefined) updateDefaults[k] = v.default;
      }
    }
    setUpdateValues(updateDefaults);
  }, []);

  const selectedEntry = selectedName ? plots[selectedName] : null;

  const handleSelectPlot = useCallback(
    (name: string) => {
      setSelectedName(name);
      const entry = plots[name];
      if (entry) _resetFormValues(entry);
    },
    [plots, _resetFormValues]
  );

  const handleOpen = useCallback(() => {
    if (!selectedName || !selectedEntry) return;
    const id = `${selectedName}-${Date.now()}`;
    setOpenedPlots((prev) => [
      ...prev,
      {
        id,
        name: selectedName,
        initParams: { ...initValues },
        updateParams: { ...updateValues },
      },
    ]);
  }, [selectedName, selectedEntry, initValues, updateValues]);

  const handleClose = useCallback((id: string) => {
    setOpenedPlots((prev) => prev.filter((p) => p.id !== id));
  }, []);

  if (loading) {
    return <div style={{ padding: 20 }}>Loading available plots…</div>;
  }
  if (error) {
    return <div style={{ padding: 20, color: 'red' }}>Error: {error}</div>;
  }

  const plotNames = Object.keys(plots);

  return (
    <div>
      <div style={{ display: 'flex', gap: 20 }}>
        {/* Left: plot list */}
        <div style={{ minWidth: 200 }}>
          <h3 style={{ marginTop: 0 }}>Available Plots</h3>
          {plotNames.length === 0 && <p style={{ color: '#888' }}>No plots found.</p>}
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {plotNames.map((name) => (
              <li
                key={name}
                onClick={() => handleSelectPlot(name)}
                style={{
                  padding: '8px 12px',
                  cursor: 'pointer',
                  borderRadius: 4,
                  marginBottom: 4,
                  background: name === selectedName ? '#007bff' : '#f5f5f5',
                  color: name === selectedName ? 'white' : '#333',
                }}
              >
                <strong>{name}</strong>
                <br />
                <span style={{ fontSize: '0.85em', opacity: 0.8 }}>
                  {plots[name].description}
                </span>
              </li>
            ))}
          </ul>
        </div>

        {/* Right: parameter forms + launch */}
        {selectedEntry && (
          <div style={{ flex: 1 }}>
            <h3 style={{ marginTop: 0 }}>{selectedName}</h3>
            <p style={{ color: '#666' }}>{selectedEntry.description}</p>

            {/* Init params */}
            {selectedEntry.parameters?.properties &&
              Object.keys(selectedEntry.parameters.properties).length > 0 && (
                <fieldset
                  style={{
                    marginBottom: 16,
                    border: '1px solid #ddd',
                    borderRadius: 4,
                    padding: 12,
                  }}
                >
                  <legend>
                    <strong>Init Parameters</strong>
                  </legend>
                  <SchemaForm
                    schema={selectedEntry.parameters.properties}
                    values={initValues}
                    onChange={(name, value) =>
                      setInitValues((prev) => ({ ...prev, [name]: value }))
                    }
                  />
                </fieldset>
              )}

            {/* Update params */}
            {selectedEntry.update_schema?.properties &&
              Object.keys(selectedEntry.update_schema.properties).length > 0 && (
                <fieldset
                  style={{
                    marginBottom: 16,
                    border: '1px solid #ddd',
                    borderRadius: 4,
                    padding: 12,
                  }}
                >
                  <legend>
                    <strong>Update Parameters</strong>
                  </legend>
                  <SchemaForm
                    schema={selectedEntry.update_schema.properties}
                    values={updateValues}
                    onChange={(name, value) =>
                      setUpdateValues((prev) => ({ ...prev, [name]: value }))
                    }
                  />
                </fieldset>
              )}

            <button
              onClick={handleOpen}
              style={{
                padding: '10px 24px',
                background: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 16,
              }}
            >
              Open Plot
            </button>
          </div>
        )}
      </div>

      {/* Opened plots */}
      {openedPlots.length > 0 && (
        <div style={{ marginTop: 30 }}>
          <h3>Open Figures</h3>
          {openedPlots.map((op) => (
            <div
              key={op.id}
              style={{
                marginBottom: 20,
                border: '1px solid #ddd',
                borderRadius: 8,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '8px 12px',
                  background: '#f5f5f5',
                }}
              >
                <strong>{op.name}</strong>
                <button
                  onClick={() => handleClose(op.id)}
                  style={{
                    padding: '4px 12px',
                    background: '#dc3545',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer',
                  }}
                >
                  Close
                </button>
              </div>
              <MatplotlibPlot
                plotName={op.name}
                baseUrl={baseUrl}
                token={token}
                initParams={op.initParams}
                updateParams={op.updateParams}
                showToolbar={true}
                showUpdateForm={true}
                style={{ minHeight: 450, padding: 10 }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
