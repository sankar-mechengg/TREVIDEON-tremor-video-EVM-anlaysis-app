import { useEffect, useRef, useState } from 'react'
import Hand3DViewer, { autoAlpha, FCOL, FINGERS, useViewerControls } from '../components/Hand3DViewer.jsx'

const BASE = '/examples/p06-high-3d'
const um = (mm) => (mm == null ? '–' : `${(mm * 1000).toFixed(0)} µm`)

/** Home-page 3D example: the segmented hand next to the reconstructed 3D hand, frames and magnified tremor. */
export default function Example3D() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(false)
  const [time, setTime] = useState(0)
  const [ctl, ctlApi, setCtl] = useViewerControls({ mode: 'frozen', alpha: 150, show: { world: false, wrist: true, fingers: true, head: false, trails: true, ellipsoids: true } })
  const vid = useRef(null)

  useEffect(() => {
    fetch(`${BASE}/kinematics.json`).then((r) => (r.ok ? r.json() : Promise.reject())).then((d) => {
      setData(d)
      setCtl((c) => ({ ...c, alpha: autoAlpha(d, 'hybrid') }))
    }).catch(() => setErr(true))
  }, [])
  useEffect(() => {
    let raf
    const tick = () => { if (vid.current) setTime(vid.current.currentTime); raf = requestAnimationFrame(tick) }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  if (err) return null
  const mh = data?.metrics?.hybrid
  const tip = (f) => mh ? Math.sqrt(['x', 'y', 'z'].reduce((s, a) => s + (mh[f]?.tip?.[a]?.band_rms || 0) ** 2, 0)) : null
  return (
    <div className="ex3d">
      <div className="ex3d-panes">
        <div className="ex3d-pane">
          <video ref={vid} src={`${BASE}/segmented.mp4`} muted playsInline autoPlay loop preload="auto" />
          <span className="ex3d-label">Segmented hand (original camera view)</span>
        </div>
        <div className="ex3d-pane">
          {data ? <Hand3DViewer data={data} time={time} controls={ctl} hasVibeSkin={false} /> : <div className="ex3d-loading">Loading 3D hand…</div>}
          <span className="ex3d-label">3D hand · drag to orbit · scroll to zoom</span>
        </div>
      </div>
      <div className="ex3d-bar">
        <div className="seg">
          <button className={`seg-btn ${ctl.mode === 'frozen' && ctl.alpha > 1 ? 'on' : ''}`}
            onClick={() => setCtl((c) => ({ ...c, mode: 'frozen', alpha: autoAlpha(data, 'hybrid'), show: { ...c.show, trails: true, ellipsoids: true } }))}>Finger tremor ×{data ? autoAlpha(data, 'hybrid') : '…'}, wrist frozen</button>
          <button className={`seg-btn ${ctl.mode === 'follow' && ctl.alpha === 1 ? 'on' : ''}`}
            onClick={() => setCtl((c) => ({ ...c, mode: 'follow', alpha: 1, show: { ...c.show, trails: false, ellipsoids: false } }))}>Measured motion</button>
        </div>
        <label className="chk"><input type="checkbox" checked={ctl.show.fingers} onChange={() => ctlApi.toggle('fingers')} /> finger frames</label>
        <label className="chk"><input type="checkbox" checked={ctl.show.wrist} onChange={() => ctlApi.toggle('wrist')} /> wrist frame</label>
      </div>
      {data && (
        <div className="ex3d-nums">
          {FINGERS.map((f) => (
            <div key={f}><span className="dot" style={{ background: FCOL[f] }} /> <span className="cap">{f}</span> <b>{um(tip(f))}</b></div>
          ))}
          <div className="dim">fingertip tremor relative to the wrist (hybrid, {data.band[0]}–{data.band[1]} Hz band RMS) · {data.model === 'wilor' ? 'WiLoR + MANO' : 'MediaPipe'} · {data.fps.toFixed(0)} fps</div>
        </div>
      )}
    </div>
  )
}
