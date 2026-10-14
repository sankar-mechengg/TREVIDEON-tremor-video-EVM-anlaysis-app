import { useCallback, useEffect, useRef, useState } from 'react'
import { fmtTime } from '../api.js'

const RATES = [0.1, 0.25, 0.5, 1, 2]

/**
 * Two frame-synchronised <video> panes with shared transport, synced zoom/pan.
 * The left video is the master clock; the right follows it.
 */
export default function ComparePlayer({
  left, right, fps = 30, onTime, seekRequest, rightPlaceholder, rightTabs, leftExtra, initialTime = 0, rightNode = null,
}) {
  const L = useRef(null)
  const R = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [t, setT] = useState(0)
  const [dur, setDur] = useState(0)
  const [rate, setRate] = useState(1)
  const [loop, setLoop] = useState(true)
  const [view, setView] = useState({ s: 1, x: 0, y: 0 })
  const drag = useRef(null)
  const leftPane = useRef(null)
  const rightPane = useRef(null)
  const hasRight = !!right?.src

  // master clock loop
  useEffect(() => {
    let raf
    const tick = () => {
      const l = L.current
      if (l) {
        setT(l.currentTime)
        onTime?.(l.currentTime)
        const r = R.current
        if (r && hasRight && r.readyState >= 2) {
          const d = r.currentTime - l.currentTime
          if (!l.paused) {
            if (Math.abs(d) > 0.12) r.currentTime = l.currentTime
            else if (Math.abs(d) > 0.03) r.playbackRate = rate * (d > 0 ? 0.95 : 1.05)
            else if (r.playbackRate !== rate) r.playbackRate = rate
            if (r.paused) r.play().catch(() => {})
          } else if (Math.abs(d) > 0.01) {
            r.currentTime = l.currentTime
          }
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [hasRight, rate, onTime])

  useEffect(() => { setPlaying(false); setT(0) }, [left?.src])

  useEffect(() => {
    if (!seekRequest || !L.current) return
    L.current.currentTime = seekRequest.t
    if (R.current) R.current.currentTime = seekRequest.t
  }, [seekRequest])

  const play = useCallback(() => {
    const l = L.current
    if (!l) return
    if (R.current && hasRight) R.current.currentTime = l.currentTime
    l.play().catch(() => {})
    if (R.current && hasRight) R.current.play().catch(() => {})
    setPlaying(true)
  }, [hasRight])

  const pause = useCallback(() => {
    L.current?.pause()
    R.current?.pause()
    if (R.current && L.current) R.current.currentTime = L.current.currentTime
    setPlaying(false)
  }, [])

  const seek = useCallback((x) => {
    const l = L.current
    if (!l) return
    const v = Math.max(0, Math.min(x, l.duration || 0))
    l.currentTime = v
    if (R.current) R.current.currentTime = v
  }, [])

  const step = useCallback((n) => {
    pause()
    seek((L.current?.currentTime || 0) + n / fps)
  }, [pause, seek, fps])

  useEffect(() => {
    if (L.current) L.current.playbackRate = rate
    if (R.current) R.current.playbackRate = rate
  }, [rate])

  // keyboard shortcuts
  useEffect(() => {
    const h = (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return
      if (e.code === 'Space') { e.preventDefault(); playing ? pause() : play() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); step(e.shiftKey ? 10 : 1) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); step(e.shiftKey ? -10 : -1) }
      else if (e.key === '0') setView({ s: 1, x: 0, y: 0 })
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [playing, play, pause, step])

  const onEnded = () => {
    if (loop) { seek(0); play() } else setPlaying(false)
  }

  // synced zoom / pan
  const onWheel = useCallback((e) => {
    e.preventDefault()
    const rect = e.currentTarget.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    setView((v) => {
      const s2 = Math.min(10, Math.max(1, v.s * (e.deltaY < 0 ? 1.15 : 1 / 1.15)))
      if (s2 === 1) return { s: 1, x: 0, y: 0 }
      return clampView({ s: s2, x: cx - (cx - v.x) * (s2 / v.s), y: cy - (cy - v.y) * (s2 / v.s) }, rect)
    })
  }, [])

  useEffect(() => {
    const panes = rightNode ? [leftPane.current] : [leftPane.current, rightPane.current]
    panes.forEach((pane) => pane?.addEventListener('wheel', onWheel, { passive: false }))
    return () => panes.forEach((pane) => pane?.removeEventListener('wheel', onWheel))
  }, [onWheel, !!rightNode]) // eslint-disable-line react-hooks/exhaustive-deps

  const onDown = (e) => {
    if (view.s === 1) return
    drag.current = { x: e.clientX, y: e.clientY, v: view, rect: e.currentTarget.getBoundingClientRect() }
  }
  const onMove = (e) => {
    const d = drag.current
    if (!d) return
    setView(clampView({ s: d.v.s, x: d.v.x + e.clientX - d.x, y: d.v.y + e.clientY - d.y }, d.rect))
  }
  const onUp = () => { drag.current = null }
  const tf = { transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }

  const paneProps = {
    onMouseDown: onDown, onMouseMove: onMove, onMouseUp: onUp, onMouseLeave: onUp,
    onDoubleClick: () => setView({ s: 1, x: 0, y: 0 }),
    style: { cursor: view.s > 1 ? 'grab' : 'zoom-in' },
  }

  return (
    <div className="compare">
      <div className="panes">
        <div className="pane">
          <div className="pane-head">
            <span className="pane-title">{left?.label || 'Original'}</span>
            {leftExtra}
          </div>
          <div ref={leftPane} className="pane-body" {...paneProps}>
            {left?.src ? (
              <video
                key={left.src}
                ref={L}
                src={left.src}
                style={tf}
                muted
                playsInline
                preload="auto"
                onLoadedMetadata={(e) => {
                  setDur(e.currentTarget.duration)
                  if (initialTime) e.currentTarget.currentTime = initialTime
                }}
                onEnded={onEnded}
                onPause={() => setPlaying(!L.current?.paused)}
              />
            ) : <div className="placeholder">Select a video</div>}
          </div>
        </div>
        <div className="pane">
          <div className="pane-head">
            <span className="pane-title">{right?.label || 'Processed'}</span>
            {rightTabs}
          </div>
          <div ref={rightPane} className={`pane-body ${rightNode ? 'pane-3d' : ''}`} {...(rightNode ? {} : paneProps)}>
            {rightNode || (hasRight ? (
              <video key={right.src} ref={R} src={right.src} style={tf} muted playsInline preload="auto"
                onLoadedData={() => { if (L.current && R.current) R.current.currentTime = L.current.currentTime }} />
            ) : <div className="placeholder">{rightPlaceholder}</div>)}
          </div>
        </div>
      </div>

      <div className="transport">
        <button className="btn icon" title="Previous frame (←)" onClick={() => step(-1)}>⏮</button>
        <button className="btn icon primary" title="Play / pause (space)" onClick={playing ? pause : play}>
          {playing ? '❚❚' : '▶'}
        </button>
        <button className="btn icon" title="Next frame (→)" onClick={() => step(1)}>⏭</button>
        <span className="time mono">{fmtTime(t)} / {fmtTime(dur)}</span>
        <input className="scrub" type="range" min={0} max={dur || 0} step={1 / fps} value={t}
          onChange={(e) => seek(parseFloat(e.target.value))} />
        <div className="seg">
          {RATES.map((r) => (
            <button key={r} className={`seg-btn ${rate === r ? 'on' : ''}`} onClick={() => setRate(r)}>{r}×</button>
          ))}
        </div>
        <label className="chk"><input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} /> loop</label>
        <span className="zoom mono" title="Wheel to zoom, drag to pan, double-click or 0 to reset">🔍 {view.s.toFixed(1)}×</span>
      </div>
    </div>
  )
}

function clampView(v, rect) {
  const w = rect.width
  const h = rect.height
  return { s: v.s, x: Math.min(0, Math.max(w - w * v.s, v.x)), y: Math.min(0, Math.max(h - h * v.s, v.y)) }
}
