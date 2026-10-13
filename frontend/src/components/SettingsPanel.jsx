import { useState } from 'react'

function Field({ f, value, onChange, maxOverride }) {
  const changed = JSON.stringify(value) !== JSON.stringify(f.default)
  const max = maxOverride ?? f.max
  const label = (
    <div className="f-label">
      <span className={`f-dot ${changed ? 'on' : ''}`} title={changed ? `Default: ${f.default}` : 'Default'} />
      <span>{f.label}</span>
      {f.help && <span className="help" title={f.help}>?</span>}
    </div>
  )
  if (f.type === 'bool') {
    return (
      <div className="field row">
        {label}
        <button className={`toggle ${value ? 'on' : ''}`} onClick={() => onChange(!value)} aria-pressed={!!value}>
          <span />
        </button>
      </div>
    )
  }
  if (f.type === 'select') {
    return (
      <div className="field">
        {label}
        <select value={String(value)} onChange={(e) => {
          const o = f.options.find((o) => String(o[0]) === e.target.value)
          onChange(o ? o[0] : e.target.value)
        }}>
          {f.options.map(([v, l]) => <option key={String(v)} value={String(v)}>{l}</option>)}
        </select>
      </div>
    )
  }
  const slider = max - f.min <= 1000
  return (
    <div className="field">
      {label}
      <div className="num">
        {slider && (
          <input type="range" min={f.min} max={max} step={f.step} value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))} />
        )}
        <input type="number" className="mono" min={f.min} max={max} step={f.step} value={value}
          onChange={(e) => onChange(e.target.value === '' ? f.min : parseFloat(e.target.value))} />
        {f.unit && <span className="unit">{f.unit}</span>}
      </div>
    </div>
  )
}

function estimateMin(settings, v, gpu) {
  if (!v) return null
  const d = settings.duration_s > 0 ? Math.min(settings.duration_s, v.duration - settings.start_s) : Math.max(0, v.duration - settings.start_s)
  const h = settings.hand_isolation ? settings.canvas_px : (settings.work_height || v.height)
  const pix = (h / 720) ** 2 * (settings.hand_isolation ? 0.56 : 1)
  const perFrame = (gpu ? 0.12 : 0.2) * pix + (settings.hand_isolation ? (gpu ? 0.1 : 0.15) : 0)
  return Math.max(1, Math.round((d * v.fps * perFrame) / 60))
}

export default function SettingsPanel({
  schema, settings, setSettings, onReset, status, onProcess, job, onCancel, videoInfo, recommended, onAutoSegment,
  onRunAll, nVideos = 0, gpu = false, presets = [], onPreset,
}) {
  const [open, setOpen] = useState(() => Object.fromEntries(schema.map((g) => [g.group, true])))
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }))
  const busy = job && ['queued', 'running'].includes(job.state)

  let statusEl = null
  if (status) {
    if (status.stage_a_cached && status.stage_b_cached) statusEl = <div className="status ok">✓ Cached — result loads instantly</div>
    else if (status.stage_a_cached) statusEl = <div className="status warn">↻ Overlay re-render only (motion analysis cached)</div>
    else {
      const est = estimateMin(settings, videoInfo, gpu)
      statusEl = <div className="status todo">⚙ Full processing required (≈ {est} min{gpu ? ', GPU' : ''})</div>
    }
  }

  return (
    <aside className="settings">
      <div className="settings-head">
        <h3>Settings</h3>
        <button className="btn ghost sm" onClick={onReset} title="Restore recommended defaults">Reset</button>
      </div>
      {presets.length > 0 && (
        <div className="presets">
          {presets.map((p) => (
            <button key={p.id} className="preset" title={p.help} onClick={() => onPreset(p)}>{p.label}</button>
          ))}
        </div>
      )}
      <div className="settings-body">
        {schema.map((g) => (
          <section key={g.group} className={`group ${g.fields.some((f) => f.key === 'hand_isolation') ? 'iso' : ''} ${g.fields.some((f) => f.key === 'hand_isolation') && settings.hand_isolation ? 'on' : ''}`}>
            <button className="group-head" onClick={() => setOpen((o) => ({ ...o, [g.group]: !o[g.group] }))}>
              <span>{open[g.group] ? '▾' : '▸'} {g.group}</span>
              {g.fields[0].stage === 'overlay' && <span className="badge">fast re-render</span>}
            </button>
            {open[g.group] && (
              <div className="group-body">
                {g.group === 'Segment' && recommended && (
                  <div className="reco">
                    <span>Suggested: <b className="mono">{recommended.start_s.toFixed(1)}s</b> ({recommended.visible_pct ?? '–'}% hand visible, steadiest)</span>
                    <button className="btn sm" onClick={onAutoSegment}>Use</button>
                  </div>
                )}
                {g.fields.filter((f) => !f.requires || settings[f.requires]).map((f) => (
                  <Field key={f.key} f={f} value={settings[f.key]} onChange={(v) => set(f.key, v)}
                    maxOverride={f.key === 'start_s' && videoInfo ? Math.floor(videoInfo.duration)
                      : f.fps_limit && videoInfo ? videoInfo.max_band_hz : undefined} />
                ))}
              </div>
            )}
          </section>
        ))}
      </div>
      <div className="settings-foot">
        {statusEl}
        {busy ? (
          <div className="jobbox">
            <div className="job-line">
              <span>{job.message || job.stage}</span>
              <span className="mono">{Math.round((job.progress || 0) * 100)}%</span>
            </div>
            <div className="bar"><div style={{ width: `${(job.progress || 0) * 100}%` }} /></div>
            <div className="job-line dim">
              <span>{job.elapsed_s != null ? `elapsed ${Math.round(job.elapsed_s)}s` : ''}</span>
              <span>{job.eta_s != null ? `ETA ${Math.round(job.eta_s)}s` : ''}</span>
            </div>
            <button className="btn danger full" onClick={onCancel}>Cancel</button>
          </div>
        ) : (
          <>
            <button id="tour-analyse" className="btn primary full lg" onClick={onProcess} disabled={!status}>
              {status?.stage_a_cached && status?.stage_b_cached ? 'Load result' : 'Analyse this video'}
            </button>
            {onRunAll && nVideos > 1 && (
              <button className="btn full" onClick={onRunAll} title="Queue every video in the list with these settings (each keeps its own segment)">
                Run all ({nVideos} videos)
              </button>
            )}
          </>
        )}
        {job?.state === 'error' && <div className="status err" title={job.message}>✕ {job.message}</div>}
      </div>
    </aside>
  )
}
