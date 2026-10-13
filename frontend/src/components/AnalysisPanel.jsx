import { useMemo, useState } from 'react'
import { ANCHOR_COLORS } from '../api.js'
import { LineChart, Spectrogram, XYTraces } from './Charts.jsx'

const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']

export default function AnalysisPanel({ analysis, overlay, time, onSeek, markerMetrics }) {
  const [specMode, setSpecMode] = useState('velocity')
  const [specAnchor, setSpecAnchor] = useState('index')
  const [hidden, setHidden] = useState({})
  const a = analysis
  const anchors = a.anchors
  const band = a.band
  const hm = a.metrics.hand

  const spectrum = useMemo(() => {
    const f = a.spectrum.freqs
    return anchors.filter((k) => !hidden[k]).map((k) => ({
      name: k, color: ANCHOR_COLORS[k], x: f,
      y: a.spectrum.psd[k].map((p, i) => (specMode === 'velocity' ? p * (2 * Math.PI * f[i]) ** 2 : p)),
      width: k === 'dorsum' ? 1 : 1.6, dash: k === 'dorsum' ? '4 3' : undefined,
    }))
  }, [a, anchors, specMode, hidden])

  const envelope = useMemo(() => {
    const keys = [...anchors, 'hand'].filter((k) => !hidden[k])
    return keys.map((k) => ({
      name: k === 'hand' ? 'whole hand' : k, color: ANCHOR_COLORS[k], x: a.envelope.t, y: a.envelope.series[k],
      width: k === 'hand' ? 2.4 : 1.2, dash: k === 'dorsum' ? '4 3' : undefined,
    }))
  }, [a, anchors, hidden])

  const peakFinger = FINGERS.reduce((b, k) => (a.metrics[k].rms_px > (a.metrics[b]?.rms_px ?? -1) ? k : b), 'thumb')

  return (
    <div className="analysis">
      <div className="cards">
        <Card label="Dominant tremor frequency" value={`${hm.peak_hz?.toFixed(2)} Hz`} sub={`band ${band[0]}–${band[1]} Hz`} accent />
        <Card label="Whole-hand tremor RMS" value={`${hm.rms_px.toFixed(3)} px`} sub={`${hm.rms_rel_hand_pct.toFixed(3)} % of hand size`} />
        <Card label="Tremor-band power" value={`${hm.band_power_pct.toFixed(1)} %`} sub="of motion > 1 Hz (fingers)" />
        <Card label="Most tremulous finger" value={peakFinger} sub={`${a.metrics[peakFinger].rms_px.toFixed(3)} px RMS @ ${a.metrics[peakFinger].peak_hz} Hz`} color={ANCHOR_COLORS[peakFinger]} />
        <Card label="Hand tracked" value={`${a.hand_detected_pct.toFixed(0)} %`} sub={`hand size ${a.hand_size_px}px · ${a.size[0]}×${a.size[1]}`} />
        {a.isolation ? (
          <Card label="Hand isolation" value={`${a.isolation.segmenter}`}
            sub={`wrist pinned < ${a.isolation.anchor_cutoff_hz} Hz · hand ${a.isolation.hand_len_src_px.toFixed(0)}→${a.isolation.hand_len_px} px · ${a.processing_s}s`} />
        ) : (
          <Card label="Camera correction" value={a.stabilization?.correction_rms_px != null ? `${a.stabilization.correction_rms_px.toFixed(2)} px` : 'off'} sub={`${a.params.stabilization} · processed in ${a.processing_s}s`} />
        )}
      </div>
      {(a.frame_repair?.repaired_frames > 0 || a.isolation || a.fps < 99) && (
        <div className="notes-strip">
          {a.isolation && <span>🖐 Wrist-anchored, background removed: amplitudes are in normalised-hand pixels (hand length {a.isolation.hand_len_px} px); whole-hand RMS = {(hm.rms_pct_hand_len ?? 0).toFixed(3)} % of hand length.</span>}
          {a.frame_repair?.repaired_frames > 0 && <span>🎞 {a.frame_repair.repaired_frames} repeated frames ({a.frame_repair.repaired_pct}%) were replaced by motion-interpolated frames before measuring.</span>}
          {(a.frame_repair?.suspected_dropped_pct ?? 0) + (a.frame_repair?.repaired_pct ?? 0) > 4 && (
            <span className="warn-line">⚠ Irregular frame timing (repeated{a.frame_repair?.suspected_dropped_frames != null ? ` and ~${a.frame_repair.suspected_dropped_frames} dropped` : ''} frames), typical of screen / VR-headset recordings. Spectral peaks at regular spacing (e.g. every 2.5 Hz) may come from the frame timing rather than the hand. Use a camera with a constant ≥100 fps for reliable frequencies.</span>
          )}
          {a.fps < 99 && <span>⚠ {a.fps.toFixed(0)} fps video: frequencies above {(a.fps * 0.45).toFixed(1)} Hz cannot be measured; ≥ 100 fps is recommended.</span>}
        </div>
      )}

      <div className="chips">
        {[...anchors, 'hand'].map((k) => (
          <button key={k} className={`chip ${hidden[k] ? 'off' : ''}`} onClick={() => setHidden((h) => ({ ...h, [k]: !h[k] }))}>
            <span className="dot" style={{ background: ANCHOR_COLORS[k] }} />{k === 'hand' ? 'whole hand' : k}
          </button>
        ))}
      </div>

      <div className="grid2">
        <div className="panel">
          <div className="panel-head">
            <h4>Tremor spectrum</h4>
            <div className="seg small">
              <button className={`seg-btn ${specMode === 'velocity' ? 'on' : ''}`} onClick={() => setSpecMode('velocity')}>velocity</button>
              <button className={`seg-btn ${specMode === 'displacement' ? 'on' : ''}`} onClick={() => setSpecMode('displacement')}>displacement</button>
            </div>
          </div>
          <LineChart series={spectrum} logY xDomain={[1, a.fps / 2]} band={band} height={240}
            xLabel="frequency (Hz)" yLabel={specMode === 'velocity' ? 'PSD (px²/s²/Hz)' : 'PSD (px²/Hz)'} legend={false} />
          <p className="note">Welch PSD of Lagrangian fingertip motion (high-passed 1 Hz). Shaded = analysis band.</p>
        </div>
        <div className="panel">
          <div className="panel-head"><h4>Tremor envelope over time</h4><span className="dim">click to seek</span></div>
          <LineChart series={envelope} xDomain={[0, a.duration_s]} playhead={time} onPickX={onSeek} height={240}
            xLabel="time in segment (s)" yLabel="RMS amplitude (px)" legend={false} />
          <p className="note">Sliding RMS (0.5 s) of band-passed displacement; “whole hand” is the Eulerian field average.</p>
        </div>
        <div className="panel">
          <div className="panel-head">
            <h4>Spectrogram</h4>
            <select value={specAnchor} onChange={(e) => setSpecAnchor(e.target.value)}>
              {anchors.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </div>
          <Spectrogram t={a.spectrogram.t} f={a.spectrogram.f} db={a.spectrogram.db[specAnchor]} playhead={time}
            band={band} onPickX={onSeek} height={220} />
          <p className="note">Short-time displacement power (dB, 2 s window). Tremor appears as a horizontal ridge.</p>
        </div>
        <div className="panel">
          <div className="panel-head"><h4>Fingertip micro-trajectories</h4><span className="dim">synced to video</span></div>
          <XYTraces trace={a.trace} anchors={anchors} colors={ANCHOR_COLORS} playhead={time} window={1.0} size={104} />
          <table className="mtable">
            <thead><tr><th>anchor</th><th>peak Hz</th><th>RMS px</th><th>P95 px</th><th>band %</th></tr></thead>
            <tbody>
              {anchors.map((k) => (
                <tr key={k}>
                  <td><span className="dot" style={{ background: ANCHOR_COLORS[k] }} />{k}</td>
                  <td className="mono">{a.metrics[k].peak_hz?.toFixed(2)}</td>
                  <td className="mono">{a.metrics[k].rms_px.toFixed(3)}</td>
                  <td className="mono">{a.metrics[k].p95_px.toFixed(3)}</td>
                  <td className="mono">{a.metrics[k].band_power_pct.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {overlay && <p className="note">Overlay colour scale: 0 – {overlay.scale_E_px.toFixed(2)} px RMS (99th pct).</p>}
        </div>
      </div>
      {markerMetrics && markerMetrics.length > 0 && (
        <div className="panel">
          <div className="panel-head"><h4>Tremor per marker</h4><span className="dim">segments: whole span · events: ±1 s window · only the analysed part counts</span></div>
          <table className="mtable wide">
            <thead><tr><th>marker</th><th>value</th><th>kind</th><th>time (s)</th><th>analysed (s)</th><th>fingertip RMS px</th><th>% hand</th><th>envelope px</th><th>peak Hz</th></tr></thead>
            <tbody>
              {markerMetrics.map((m, i) => (
                <tr key={i} className={m.analysed_s > 0 ? '' : 'dim'}>
                  <td>{m.marker}</td><td>{m.value}</td><td>{m.kind}</td>
                  <td className="mono">{m.time_s}{m.end_s != null ? `–${m.end_s}` : ''}</td>
                  <td className="mono">{m.analysed_s}</td>
                  <td className="mono">{m.fingertip_rms_px ?? '–'}</td>
                  <td className="mono">{m.rms_pct_hand_len ?? '–'}</td>
                  <td className="mono">{m.hand_envelope_px ?? '–'}</td>
                  <td className="mono">{m.peak_hz ?? '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function Card({ label, value, sub, accent, color }) {
  return (
    <div className={`card ${accent ? 'accent' : ''}`}>
      <div className="card-label">{label}</div>
      <div className="card-value" style={color ? { color } : undefined}>{value}</div>
      <div className="card-sub">{sub}</div>
    </div>
  )
}
