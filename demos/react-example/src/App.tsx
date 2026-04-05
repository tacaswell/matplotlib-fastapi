import React, { useState, useMemo } from 'react';
import { MatplotlibPlot } from './MatplotlibPlot';
import { PlotPicker } from './PlotPicker';

export default function App() {
  const [phase, setPhase] = useState(0.0);
  const [tab, setTab] = useState<'picker' | 'direct'>('picker');

  // Memoize updateParams to avoid unnecessary re-renders
  const updateParams = useMemo(() => ({ phase }), [phase]);

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>
      <h1>🎨 mpl-fastapi React Example</h1>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        <button
          onClick={() => setTab('picker')}
          style={{
            padding: '8px 20px',
            background: tab === 'picker' ? '#007bff' : '#e9ecef',
            color: tab === 'picker' ? 'white' : '#333',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          Plot Picker
        </button>
        <button
          onClick={() => setTab('direct')}
          style={{
            padding: '8px 20px',
            background: tab === 'direct' ? '#007bff' : '#e9ecef',
            color: tab === 'direct' ? 'white' : '#333',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          Direct Usage Example
        </button>
      </div>

      {tab === 'picker' && (
        <div style={{
          background: 'white',
          padding: 20,
          borderRadius: 8,
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}>
          <h2 style={{ marginTop: 0 }}>Plot Picker</h2>
          <p style={{ color: '#666' }}>
            Discover all available plots on the server.  Select one, fill in
            parameters, and click <strong>Open Plot</strong> to view it.
          </p>
          <PlotPicker baseUrl="/plots" />
        </div>
      )}

      {tab === 'direct' && (
        <>
          <div style={{
            background: 'white',
            padding: 20,
            borderRadius: 8,
            marginBottom: 20,
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h2>Phase Control</h2>
            <p style={{ color: '#666', marginBottom: 15 }}>
              Drag the slider to update the sine wave phase in real-time via WebSocket.
            </p>
            <label style={{ display: 'flex', alignItems: 'center', gap: 15 }}>
              <span style={{ minWidth: 100 }}>Phase: {phase.toFixed(2)} rad</span>
              <input
                type="range"
                min="0"
                max="6.28"
                step="0.05"
                value={phase}
                onChange={(e) => setPhase(parseFloat(e.target.value))}
                style={{ flex: 1, maxWidth: 300 }}
              />
            </label>
          </div>

          <div style={{
            background: 'white',
            padding: 20,
            borderRadius: 8,
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h2>Interactive Sine Plot</h2>
            <MatplotlibPlot
              plotName="interactive_sine"
              baseUrl="/plots"
              initParams={{ frequency: 2.0, amplitude: 1.5 }}
              updateParams={updateParams}
              showToolbar={true}
              showUpdateForm={false}
              style={{ minHeight: 450, border: '1px solid #ddd', borderRadius: 4 }}
            />
          </div>

          <div style={{ marginTop: 20, color: '#666', fontSize: 14 }}>
            <strong>How it works:</strong>
            <ul>
              <li>React app built with Vite, outputs to <code>demos/react-example/dist/</code></li>
              <li>FastAPI serves the built React app at <code>/react-app/</code></li>
              <li>WebSocket connects to <code>/plots/ws/v0/interactive_sine</code></li>
              <li>Phase updates are sent via WebSocket without reconnecting</li>
              <li>Single server handles everything - no CORS issues!</li>
            </ul>
            <p style={{ marginTop: 10 }}>
              <strong>Note:</strong> <code>initParams</code> (frequency, amplitude) are set at connection time.
              Only <code>updateParams</code> (phase) can be changed without reconnecting.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
