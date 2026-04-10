import React, { useEffect, useState } from 'react';
import { VERSION } from 'mpl-fastapi';

interface WatermarkProps {
  baseUrl?: string;
}

export function Watermark({ baseUrl = '/plots' }: WatermarkProps) {
  const [versions, setVersions] = useState<Record<string, string> | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${baseUrl}/watermark`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (!cancelled) {
          data.mpl_fastapi_js = VERSION;
          setVersions(data);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setVersions({ mpl_fastapi_js: VERSION });
        }
      });
    return () => { cancelled = true; };
  }, [baseUrl]);

  if (!versions) return null;

  return (
    <div
      style={{
        marginTop: 30,
        padding: '8px 12px',
        background: '#e9ecef',
        borderRadius: 4,
        fontSize: 11,
        color: '#888',
        fontFamily: 'monospace',
        textAlign: 'center',
      }}
    >
      {Object.entries(versions)
        .map(([k, v]) => `${k}: ${v}`)
        .join(' | ')}
    </div>
  );
}
