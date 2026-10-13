import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api, fmtBytes } from '../api.js'

/* ------------------------------------------------------------------ source: local folder */
export function LibraryBar({ library, onChanged, onError }) {
  const [busy, setBusy] = useState(false)
  const browse = async (kind) => {
    setBusy(true)
    try {
      const r = await api.browse(kind)
      if (!r.cancelled) onChanged(r)
    } catch (e) { onError(e.message) } finally { setBusy(false) }
  }
  const pick = async (dir) => {
    try { onChanged(await api.setLibrary(dir)) } catch (e) { onError(e.message) }
  }
  if (!library) return null
  return (
    <div className="libbar">
      <div className="lib-row">
        <span className="lib-label">Videos</span>
        <span className="lib-path mono" title={library.data_dir}>{library.data_dir}</span>
      </div>
      <div className="lib-row">
        <span className="lib-label">Results</span>
        <span className="lib-path mono" title={library.results_dir}>{library.results_dir}</span>
      </div>
      <div className="lib-actions">
        <button className="btn sm primary" disabled={busy} onClick={() => browse('data')}>{busy ? 'Waiting for the dialog…' : 'Browse…'}</button>
        <button className="btn sm" disabled={busy} onClick={() => browse('results')} title="Choose another results folder">Results folder…</button>
        {library.recent?.length > 1 && (
          <select value="" onChange={(e) => e.target.value && pick(e.target.value)} title="Recent folders">
            <option value="">Recent…</option>
            {library.recent.slice(1).map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ source: hosted uploads */
export function UploadPanel({ limits, onUploaded, onError }) {
  const [items, setItems] = useState([])
  const [drag, setDrag] = useState(false)
  const input = useRef(null)
  const maxB = (limits?.max_upload_mb || 1024) * 1024 * 1024

  const add = async (files) => {
    const vids = files.filter((f) => /\.(mp4|mov|m4v|mkv|avi)$/i.test(f.name))
    const csvs = files.filter((f) => /\.csv$/i.test(f.name))
    for (const f of vids) {
      const id = `${f.name}-${f.size}-${Math.random()}`
      if (f.size > maxB) {
        setItems((it) => [...it, { id, name: f.name, size: f.size, state: 'error', msg: `larger than ${fmtBytes(maxB)}` }])
        continue
      }
      setItems((it) => [...it, { id, name: f.name, size: f.size, state: 'uploading', p: 0 }])
      try {
        const r = await api.upload(f, (p) => setItems((it) => it.map((x) => (x.id === id ? { ...x, p } : x))))
        const stem = f.name.replace(/\.[^.]+$/, '')
        const csv = csvs.find((c) => c.name.replace(/\.[^.]+$/, '').replace(/[_-]?markers?$/i, '') === stem)
        let mk = null
        if (csv) mk = await api.uploadMarkers(r.video, await csv.text())
        setItems((it) => it.map((x) => (x.id === id ? { ...x, state: 'done', video: r.video, fps: r.fps, fps_status: r.fps_status,
          msg: mk ? (mk.errors.length ? `markers: ${mk.errors.length} error(s)` : `markers: ${mk.rows.length} rows`) : '' } : x)))
        onUploaded(r.video)
      } catch (e) {
        setItems((it) => it.map((x) => (x.id === id ? { ...x, state: 'error', msg: e.message } : x)))
        onError?.(e.message)
      }
    }
  }

  return (
    <div className="upload">
      <div className={`dropzone ${drag ? 'drag' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); add([...e.dataTransfer.files]) }}
        onClick={() => input.current?.click()} role="button" tabIndex={0}>
        <div className="dz-icon">⬆</div>
        <b>Drop MP4 videos here</b>
        <span className="dim">or click to choose · up to {fmtBytes(maxB)} each · several at once</span>
        <span className="dim">Optional: drop a marker CSV named like the video (e.g. <code>trial1.csv</code> for <code>trial1.mp4</code>)</span>
        <input ref={input} type="file" multiple accept="video/*,.mp4,.mov,.mkv,.avi,.csv" hidden
          onChange={(e) => { add([...e.target.files]); e.target.value = '' }} />
      </div>
      {items.length > 0 && (
        <ul className="uplist">
          {items.map((x) => (
            <li key={x.id} className={x.state}>
              <span className="up-name" title={x.name}>{x.name}</span>
              <span className="dim">{fmtBytes(x.size)}</span>
              {x.state === 'uploading' && <div className="bar"><div style={{ width: `${(x.p || 0) * 100}%` }} /></div>}
              {x.state === 'done' && <span className={`fps ${x.fps_status === 'ok' ? 'ok' : 'warn'}`}>{x.fps?.toFixed(0)} fps</span>}
              {x.msg && <span className={x.state === 'error' ? 'err' : 'dim'}>{x.msg}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ markers */
export function MarkersPanel({ video, markers, onChange }) {
  const [msg, setMsg] = useState(null)
  const input = useRef(null)
  if (!video) return null
  const upload = async (file) => {
    try {
      const r = await api.uploadMarkers(video, await file.text())
      setMsg(r)
      onChange()
    } catch (e) { setMsg({ errors: [e.message], warnings: [], rows: [] }) }
  }
  const remove = async () => { await api.deleteMarkers(video); setMsg(null); onChange() }
  const rows = markers?.rows || []
  return (
    <div className="panel markers" id="tour-markers">
      <div className="panel-head">
        <h4>Markers <span className="dim">(optional)</span></h4>
        <div className="row-btns">
          <a className="btn sm ghost" href={api.templateUrl()} download>Template</a>
          <button className="btn sm" onClick={() => input.current?.click()}>{rows.length ? 'Replace CSV' : 'Upload CSV'}</button>
          {markers?.present && <button className="btn sm ghost" onClick={remove}>Remove</button>}
          <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => { e.target.files[0] && upload(e.target.files[0]); e.target.value = '' }} />
        </div>
      </div>
      {msg?.errors?.length > 0 && <div className="status err">✕ {msg.errors.slice(0, 6).join(' · ')}</div>}
      {(msg?.warnings || markers?.warnings || []).length > 0 && <div className="status warn">⚠ {(msg?.warnings || markers.warnings).slice(0, 4).join(' · ')}</div>}
      {rows.length ? (
        <p className="dim">{rows.length} markers: {[...new Set(rows.map((r) => r.marker))].join(', ')}. Shown on the timeline; tremor per marker appears with the results.</p>
      ) : (
        <p className="dim">Add a CSV with <code>time_s, end_s, marker, value, notes</code> to get tremor statistics per condition / event.</p>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ steps */
export function Stepper({ steps, onStep, onTour, right }) {
  return (
    <div className="stepper">
      <ol>
        {steps.map((s, i) => (
          <li key={s.id} className={`${s.state}`} onClick={() => onStep(s)}>
            <span className="st-n">{s.state === 'done' ? '✓' : i + 1}</span>
            <span className="st-l">{s.label}</span>
          </li>
        ))}
      </ol>
      <button className="btn sm ghost" onClick={onTour}>Guide me</button>
      {right}
    </div>
  )
}

/* ------------------------------------------------------------------ guided tour */
export function Tour({ steps, open, onClose }) {
  const [i, setI] = useState(0)
  const [rect, setRect] = useState(null)
  useEffect(() => { if (open) setI(0) }, [open])
  useLayoutEffect(() => {
    if (!open) return
    const el = document.getElementById(steps[i]?.target)
    if (!el) { setRect(null); return }
    el.scrollIntoView({ block: 'center', behavior: 'smooth' })
    const upd = () => setRect(el.getBoundingClientRect())
    upd()
    const t = setTimeout(upd, 350)
    window.addEventListener('resize', upd)
    return () => { clearTimeout(t); window.removeEventListener('resize', upd) }
  }, [open, i, steps])
  if (!open) return null
  const s = steps[i]
  const pad = 6
  const box = rect ? { left: rect.left - pad, top: rect.top - pad, width: rect.width + 2 * pad, height: rect.height + 2 * pad } : null
  const below = rect && rect.top < window.innerHeight / 2
  const pop = rect ? {
    left: Math.min(Math.max(12, rect.left), window.innerWidth - 372),
    top: below ? Math.min(rect.bottom + 14, window.innerHeight - 220) : Math.max(12, rect.top - 200),
  } : { left: '50%', top: '40%', transform: 'translateX(-50%)' }
  return (
    <div className="tour" role="dialog" aria-label="Guided tour">
      {box ? <div className="tour-hole" style={box} /> : <div className="tour-dim" />}
      <div className="tour-pop" style={pop}>
        <div className="tour-step dim">Step {i + 1} of {steps.length}</div>
        <h4>{s.title}</h4>
        <p>{s.text}</p>
        <div className="row-btns">
          <button className="btn sm ghost" onClick={onClose}>Close</button>
          {i > 0 && <button className="btn sm" onClick={() => setI(i - 1)}>Back</button>}
          {i < steps.length - 1 ? <button className="btn sm primary" onClick={() => setI(i + 1)}>Next</button>
            : <button className="btn sm primary" onClick={onClose}>Start</button>}
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ queue */
export function QueuePanel({ jobs, onCancel, onOpen }) {
  const active = jobs.filter((j) => ['queued', 'running'].includes(j.state))
  const recent = jobs.filter((j) => !['queued', 'running'].includes(j.state)).slice(0, 6)
  if (!active.length && !recent.length) return null
  return (
    <div className="queue">
      <div className="q-head">Queue {active.length > 0 && <span className="dim">· {active.length} active</span>}</div>
      {[...active, ...recent].map((j) => (
        <div key={j.id} className={`q-item ${j.state}`}>
          <div className="q-line">
            <span className="q-name" title={j.video} onClick={() => onOpen(j)}>{j.video}{j.isolation ? ' · isolated' : ''}</span>
            <span className="mono">{j.state === 'running' ? `${Math.round((j.progress || 0) * 100)}%` : j.state === 'queued' ? `#${j.queue_position}` : j.state}</span>
          </div>
          {j.state === 'running' && <div className="bar"><div style={{ width: `${(j.progress || 0) * 100}%` }} /></div>}
          {['queued', 'running'].includes(j.state) && (
            <div className="q-line dim"><span>{j.message}</span><button className="btn sm ghost" onClick={() => onCancel(j)}>Cancel</button></div>
          )}
          {j.state === 'error' && <div className="q-line err" title={j.message}>{j.message}</div>}
        </div>
      ))}
    </div>
  )
}
