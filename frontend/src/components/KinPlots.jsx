import { useMemo, useState } from 'react'
import { url } from '../api.js'
import { LineChart, Spectrogram, useWidth } from './Charts.jsx'
import { FCOL, FINGERS } from './Hand3DViewer.jsx'

const AXC = { x: '#ff6b6b', y: '#6bcb77', z: '#4d96ff', flexion: '#ffd93d', abduction: '#22d3ee', twist: '#c77dff',
  x_img: '#ff6b6b', y_img: '#6bcb77', roll: '#c77dff' }
const METHOD_LABEL = { hybrid: 'Hybrid (optical flow in the wrist frame)', model: '3D model only', mediapipe: 'MediaPipe 3D (comparison)' }
const f3 = (v, d = 3) => (v == null || !isFinite(v) ? '–' : Number(v).toFixed(d))
const um = (mm) => (mm == null ? '–' : mm < 1 ? `${(mm * 1000).toFixed(0)} µm` : `${mm.toFixed(2)} mm`)

function groupsFor(body) {
  return body === 'wrist' ? { pos: 'Position (world)', rot: 'Rotation (about wrist axes)' }
    : { tip: 'Fingertip in the wrist frame', ang: 'Finger orientation relative to the wrist' }
}

export default function KinPlots({ data, time, onSeek, urls }) {
  const methods = Object.keys(data.series)
  const [method, setMethod] = useState('hybrid')
  const [body, setBody] = useState('index')
  const [kind, setKind] = useState('band')
  const t = data.t
  const f = data.f
  const band = data.band
  const ser = data.series[method] || data.series.hybrid
  const grp = groupsFor(body)
  const [gPos, gAng] = Object.keys(grp)
  const sb = ser[body] || {}

  const lines = (g, field) => Object.entries(sb[g] || {}).filter(([, e]) => e[field]).map(([a, e]) => ({
    name: `${a} (${e.unit}${field === 'vel' ? '/s' : field === 'acc' ? '/s²' : ''})`, color: AXC[a] || '#ccc', x: t, y: e[field], width: 1.2,
  }))
  const psdLines = (g) => {
    const out = []
    for (const [a, e] of Object.entries(sb[g] || {})) {
      out.push({ name: a, color: AXC[a] || '#ccc', x: f, y: e.psd, width: 1.4 })
      if (e.noise) out.push({ name: `${a} noise`, color: AXC[a] || '#ccc', x: f, y: e.noise, width: 1, dash: '4 3' })
    }
    return out
  }
  const sp = data.spectrograms?.[method === 'mediapipe' ? 'model' : method]?.[body === 'wrist' ? 'wrist_pos' : body]
  const ja = body !== 'wrist' ? data.joint_angles?.[body] : null
  const metrics = data.metrics[method] || {}
  const unitPos = Object.values(sb[gPos] || {})[0]?.unit || 'mm'

  return (
    <div className="kin-results">
      <Summary data={data} />
      <Notes data={data} />
      <div className="kin-controls">
        <div className="seg">
          {methods.map((m) => (
            <button key={m} className={`seg-btn ${method === m ? 'on' : ''}`} onClick={() => setMethod(m)} title={METHOD_LABEL[m]}>
              {m === 'hybrid' ? 'Hybrid' : m === 'model' ? (data.model === 'wilor' ? 'WiLoR' : 'Model') : 'MediaPipe'}
            </button>
          ))}
        </div>
        <div className="chips">
          {['wrist', ...FINGERS].map((b) => (
            <button key={b} className={`chip ${body === b ? 'on' : ''}`} onClick={() => setBody(b)}>
              <span className="dot" style={{ background: FCOL[b] }} />{b}
            </button>
          ))}
        </div>
        <div className="seg">
          <button className={`seg-btn ${kind === 'band' ? 'on' : ''}`} onClick={() => setKind('band')}>Tremor band</button>
          <button className={`seg-btn ${kind === 'raw' ? 'on' : ''}`} onClick={() => setKind('raw')}>Full signal</button>
        </div>
        <span className="dim">{METHOD_LABEL[method]}</span>
      </div>

      <div className="grid2">
        <Panel title={`${grp[gPos]} · ${kind === 'band' ? `${band[0]}–${band[1]} Hz` : 'full'}`}>
          <LineChart series={lines(gPos, kind)} playhead={time} onPickX={onSeek} xLabel="time (s)" yLabel={unitPos} height={200} />
        </Panel>
        <Panel title={`${grp[gAng]} · ${kind === 'band' ? 'band' : 'full'}`}>
          <LineChart series={lines(gAng, kind)} playhead={time} onPickX={onSeek} xLabel="time (s)" yLabel="deg" height={200} />
        </Panel>
        {method !== 'mediapipe' && (
          <>
            <Panel title="Velocity (tremor band)">
              <LineChart series={lines(gPos, 'vel')} playhead={time} onPickX={onSeek} xLabel="time (s)" yLabel={`${unitPos}/s`} height={180} />
            </Panel>
            <Panel title="Acceleration (tremor band)">
              <LineChart series={lines(gPos, 'acc')} playhead={time} onPickX={onSeek} xLabel="time (s)" yLabel={`${unitPos}/s²`} height={180} />
            </Panel>
          </>
        )}
        <Panel title="Power spectrum · dashed = noise floor" hint="Welch PSD of each axis. A peak counts only where it rises clearly above its dashed noise floor.">
          <LineChart series={psdLines(gPos)} logY band={band} xLabel="frequency (Hz)" yLabel={`${unitPos}²/Hz`} height={200} />
        </Panel>
        <Panel title="Spectrogram · principal tremor axis">
          {sp ? <Spectrogram t={sp.t} f={sp.f} db={sp.db} playhead={time} fMax={Math.min(30, data.fps / 2)} band={band} onPickX={onSeek} height={200} />
            : <div className="dim">–</div>}
        </Panel>
        {ja && (
          <Panel title="Joint flexion angles (model)">
            <LineChart series={Object.entries(ja).map(([k, v], i) => ({ name: k.toUpperCase(), color: ['#ffd93d', '#22d3ee', '#c77dff'][i], x: t, y: v, width: 1.2 }))}
              playhead={time} onPickX={onSeek} xLabel="time (s)" yLabel="deg" height={180} />
          </Panel>
        )}
        <Panel title="Fingertip tremor paths in the wrist frame (last 1 s)" hint="Band-passed fingertip displacement in the wrist frame, x-y plane (palm plane).">
          <TipPaths data={data} method={method === 'mediapipe' ? 'model' : method} time={time} />
        </Panel>
        <Panel title="Tremor amplitude per finger (band RMS, wrist frame)">
          <AmpBars data={data} />
        </Panel>
        <Panel title={`Coherence in the tremor band · ${method === 'mediapipe' ? 'model' : method}`}
          hint="Magnitude-squared coherence of the principal tremor axes, averaged over the band: 1 = the two move together, 0 = independent.">
          <CoherenceMatrix bodies={data.coherence.bodies} M={data.coherence[method === 'mediapipe' ? 'model' : method]} />
        </Panel>
      </div>

      <MetricsTable metrics={metrics} method={method} />

      {urls && (
        <div className="panel">
          <div className="panel-head"><b>Figures &amp; exports</b>
            <a className="btn sm primary" href={url(urls.bundle)} download>Download all (ZIP)</a></div>
          <div className="figs">
            {urls.figures.map((fg) => (
              <a key={fg.name} href={url(fg.url)} target="_blank" rel="noreferrer" title={fg.name}>
                <img src={url(fg.url)} alt={fg.name} loading="lazy" /><span>{fg.name.replace(/^\d+_/, '').replace(/_/g, ' ').replace('.png', '')}</span>
              </a>
            ))}
          </div>
          <div className="exports">
            {urls.exports.map((e) => <a key={e.name} className="chip" href={url(e.url)} download>{e.name}</a>)}
          </div>
        </div>
      )}
    </div>
  )
}

function Panel({ title, hint, children }) {
  return (
    <div className="panel">
      <div className="panel-head"><b>{title}</b>{hint && <span className="help" title={hint}>?</span>}</div>
      {children}
    </div>
  )
}

function Summary({ data }) {
  const mh = data.metrics.hybrid || {}
  const mm = data.metrics.model || {}
  const tipRms = (m, b) => {
    const g = m[b]?.tip
    if (!g) return null
    return Math.sqrt(['x', 'y', 'z'].reduce((s, a) => s + (g[a]?.band_rms || 0) ** 2, 0))
  }
  const fingersH = FINGERS.map((b) => tipRms(mh, b)).filter((v) => v != null)
  const medH = fingersH.length ? fingersH.sort((a, b) => a - b)[Math.floor(fingersH.length / 2)] : null
  const fingersM = FINGERS.map((b) => tipRms(mm, b)).filter((v) => v != null)
  const medM = fingersM.length ? fingersM.sort((a, b) => a - b)[Math.floor(fingersM.length / 2)] : null
  const snr = FINGERS.map((b) => mh[b]?.tip?.y?.snr_db).filter((v) => v != null)
  const snrMed = snr.length ? snr.sort((a, b) => a - b)[Math.floor(snr.length / 2)] : null
  const peak = mh.index?.tip?.y?.peak_hz
  const wr = mh.wrist?.pos ? Math.hypot(mh.wrist.pos.x_img?.band_rms || 0, mh.wrist.pos.y_img?.band_rms || 0) : null
  const w = data.world
  return (
    <div className="cards">
      <div className="card accent"><div className="card-label">Finger tremor (hybrid)</div>
        <div className="card-value">{um(medH)}</div><div className="card-sub">median fingertip band RMS, relative to the wrist</div></div>
      <div className="card"><div className="card-label">Signal / noise floor</div>
        <div className="card-value">{snrMed == null ? '–' : `${snrMed.toFixed(1)} dB`}</div><div className="card-sub">hybrid, distal axis, median of fingers</div></div>
      <div className="card"><div className="card-label">Peak (index, hybrid)</div>
        <div className="card-value">{peak == null ? '–' : `${peak} Hz`}</div><div className="card-sub">band {data.band[0]}–{data.band[1]} Hz · {data.fps.toFixed(0)} fps</div></div>
      <div className="card"><div className="card-label">Wrist tremor (hybrid)</div>
        <div className="card-value">{um(wr)}</div><div className="card-sub">in the image plane, head rotation removed</div></div>
      <div className="card"><div className="card-label">3D model</div>
        <div className="card-value">{data.model === 'wilor' ? 'WiLoR + MANO' : 'MediaPipe'}</div>
        <div className="card-sub">model-only finger signal {um(medM)} (jitter)</div></div>
      <div className="card"><div className="card-label">Hand · scale</div>
        <div className="card-value">{data.hand.physical_side}</div>
        <div className="card-sub">{data.scale.measured_hand_length_mm ? `measured ${data.scale.measured_hand_length_mm} mm` : `model ${data.scale.model_hand_length_mm} mm`} · {data.hand.side_from}</div></div>
      <div className="card"><div className="card-label">World frame</div>
        <div className="card-value">{w.floor_aligned ? 'floor-aligned' : w.mode === 'camera' ? 'camera' : 'stabilised'}</div>
        <div className="card-sub">head rotated up to {w.head_rotation_range_deg ?? '–'}° (removed below {w.camera_cutoff_hz ?? '–'} Hz); head translation not removed</div></div>
      <div className="card"><div className="card-label">Observability (wrist axes)</div>
        <div className="card-value mono" style={{ fontSize: 16 }}>x {f3(data.observability.x, 2)} · y {f3(data.observability.y, 2)} · z {f3(data.observability.z, 2)}</div>
        <div className="card-sub">1 = across the image, 0 = along the line of sight</div></div>
    </div>
  )
}

function Notes({ data }) {
  const notes = []
  const mm = data.metrics.model || {}
  const snrM = FINGERS.flatMap((b) => ['x', 'y', 'z'].map((a) => mm[b]?.tip?.[a]?.snr_db)).filter((v) => v != null)
  const medM = snrM.length ? snrM.sort((a, b) => a - b)[Math.floor(snrM.length / 2)] : null
  if (medM != null && medM < 3) notes.push(`The 3D model's own finger signal is at its noise floor (median SNR ${medM.toFixed(1)} dB): its jitter, not tremor. Use the hybrid values for tremor amplitude and frequency; the model provides the frames, orientation and scale.`)
  if (data.fps < 99) notes.push(`${data.fps.toFixed(0)} fps: rhythms above ${(data.fps / 2).toFixed(0)} Hz cannot be seen and the band is limited to ${data.band[1]} Hz.`)
  if (data.frame_repair?.repaired_frames) notes.push(`${data.frame_repair.repaired_frames} repeated frames repaired (${data.frame_repair.repaired_pct}%), ${data.frame_repair.suspected_dropped_frames} suspected dropped frames: irregular timing can still add lines every few Hz.`)
  const oz = data.observability?.z
  if (oz != null && oz < 0.5) notes.push(`The wrist z axis (back of the hand) points mostly along the line of sight (visibility ${oz.toFixed(2)}): hybrid motion along z (finger flexion) is under-measured.`)
  if (!notes.length) return null
  return <div className="notes">{notes.map((n, i) => <div key={i}>ⓘ {n}</div>)}</div>
}

function TipPaths({ data, method, time }) {
  const t = data.t
  const dt = t[1] - t[0] || 1 / 30
  const i1 = Math.max(0, Math.min(t.length - 1, Math.round(time / dt)))
  const i0 = Math.max(0, i1 - Math.round(1 / dt))
  const ser = data.series[method]
  const lim = useMemo(() => {
    let mx = 0
    for (const b of FINGERS) for (const a of ['x', 'y']) for (const v of ser[b].tip[a].band) mx = Math.max(mx, Math.abs(v))
    return mx || 1
  }, [ser])
  const size = 104
  const s = (v) => size / 2 + (v / lim) * (size / 2 - 6)
  return (
    <div className="xy-row">
      {FINGERS.map((b) => {
        const X = ser[b].tip.x.band, Y = ser[b].tip.y.band
        let d = ''
        for (let i = i0; i <= i1; i++) d += `${i > i0 ? 'L' : 'M'}${s(X[i]).toFixed(1)},${s(-Y[i]).toFixed(1)}`
        return (
          <div key={b} className="xy">
            <svg width={size} height={size}>
              <rect x={0.5} y={0.5} width={size - 1} height={size - 1} className="xy-frame" />
              <line x1={size / 2} x2={size / 2} y1={4} y2={size - 4} className="grid" />
              <line y1={size / 2} y2={size / 2} x1={4} x2={size - 4} className="grid" />
              <path d={d} fill="none" stroke={FCOL[b]} strokeWidth={1.4} />
              <circle cx={s(X[i1])} cy={s(-Y[i1])} r={3} fill={FCOL[b]} />
            </svg>
            <div className="xy-label" style={{ color: FCOL[b] }}>{b}</div>
          </div>
        )
      })}
      <div className="xy-note">±{(lim * 1000).toFixed(0)} µm<br />x → , y ↑</div>
    </div>
  )
}

function AmpBars({ data }) {
  const [ref, W] = useWidth()
  const rows = FINGERS.map((b) => {
    const r = (m) => {
      const g = data.metrics[m]?.[b]?.tip
      if (!g) return null
      return Math.sqrt(['x', 'y', 'z'].reduce((s, a) => s + (g[a]?.band_rms || 0) ** 2, 0))
    }
    const n = (m) => {
      const g = data.metrics[m]?.[b]?.tip
      if (!g || g.x?.noise_rms == null) return null
      return Math.sqrt(['x', 'y', 'z'].reduce((s, a) => s + (g[a]?.noise_rms || 0) ** 2, 0))
    }
    return { b, h: r('hybrid'), m: r('model'), hn: n('hybrid'), mn: n('model') }
  })
  const vmax = Math.max(...rows.flatMap((r) => [r.h, r.m, r.hn, r.mn].filter((v) => v != null)), 1e-6)
  const lg = (v) => Math.log10(Math.max(v, vmax * 1e-3))
  const x = (v) => 70 + ((lg(v) - lg(vmax * 1e-3)) / (lg(vmax) - lg(vmax * 1e-3) || 1)) * (W - 90)
  const H = 22
  return (
    <div ref={ref} className="chart">
      <svg width={W} height={rows.length * H * 2 + 26}>
        {rows.map((r, i) => {
          const y = 6 + i * H * 2
          return (
            <g key={r.b}>
              <text x={4} y={y + 15} className="tick" fill={FCOL[r.b]}>{r.b}</text>
              {r.h != null && <rect x={70} y={y + 2} width={Math.max(1, x(r.h) - 70)} height={H - 6} fill="#22d3ee" opacity={0.85} />}
              {r.hn != null && <line x1={x(r.hn)} x2={x(r.hn)} y1={y} y2={y + H - 2} stroke="#22d3ee" strokeDasharray="3 2" />}
              {r.m != null && <rect x={70} y={y + H} width={Math.max(1, x(r.m) - 70)} height={H - 6} fill="#a78bfa" opacity={0.7} />}
              {r.mn != null && <line x1={x(r.mn)} x2={x(r.mn)} y1={y + H - 2} y2={y + 2 * H - 4} stroke="#a78bfa" strokeDasharray="3 2" />}
              <text x={W - 4} y={y + 13} className="tick" textAnchor="end">{um(r.h)}</text>
              <text x={W - 4} y={y + H + 11} className="tick" textAnchor="end">{um(r.m)}</text>
            </g>
          )
        })}
        <text x={70} y={rows.length * H * 2 + 20} className="tick">log scale · cyan = hybrid, violet = model · dashed = noise floor</text>
      </svg>
    </div>
  )
}

function CoherenceMatrix({ bodies, M }) {
  if (!M) return <div className="dim">–</div>
  const n = bodies.length
  const c = 34
  const pad = 72
  const col = (v) => `hsl(${190 - v * 150}, 80%, ${18 + v * 42}%)`
  return (
    <svg width={pad + n * c + 4} height={pad + n * c + 4} className="coh">
      {bodies.map((b, i) => (
        <g key={b}>
          <text x={pad - 6} y={pad + i * c + c / 2 + 4} textAnchor="end" className="tick">{b.replace('wrist_', 'wrist ')}</text>
          <text transform={`translate(${pad + i * c + c / 2 + 4}, ${pad - 6}) rotate(-55)`} className="tick">{b.replace('wrist_', 'wrist ')}</text>
        </g>
      ))}
      {M.map((row, i) => row.map((v, j) => (
        <g key={`${i}-${j}`}>
          <rect x={pad + j * c} y={pad + i * c} width={c - 1} height={c - 1} fill={col(v ?? 0)} />
          <text x={pad + j * c + c / 2} y={pad + i * c + c / 2 + 4} textAnchor="middle" className="tick" fill={v > 0.6 ? '#000' : '#e2e8f0'}>{v == null ? '' : v.toFixed(2)}</text>
        </g>
      )))}
    </svg>
  )
}

function MetricsTable({ metrics, method }) {
  const rows = []
  for (const [b, bd] of Object.entries(metrics)) {
    for (const [g, gd] of Object.entries(bd)) {
      for (const [a, m] of Object.entries(gd)) rows.push({ b, g, a, ...m })
    }
  }
  return (
    <div className="panel">
      <div className="panel-head"><b>Metrics · {method}</b><span className="dim">band RMS = RMS of the band-passed signal; SNR against the noise floor</span></div>
      <div className="table-wrap">
        <table className="mtable">
          <thead><tr><th>body</th><th>signal</th><th>axis</th><th>peak Hz</th><th>band RMS</th><th>total RMS</th><th>band %</th><th>noise RMS</th><th>SNR dB</th><th>visibility</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td style={{ color: FCOL[r.b] }}>{r.b}</td><td>{r.g}</td><td>{r.a}</td><td className="mono">{r.peak_hz ?? '–'}</td>
                <td className="mono">{f3(r.band_rms, 4)} {r.unit}</td><td className="mono">{f3(r.total_rms, 3)}</td>
                <td className="mono">{r.band_power_pct ?? '–'}</td><td className="mono">{f3(r.noise_rms, 4)}</td>
                <td className={`mono ${r.snr_db != null ? (r.snr_db > 6 ? 'good' : r.snr_db < 2 ? 'bad' : '') : ''}`}>{r.snr_db ?? '–'}</td>
                <td className="mono">{r.observability ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
