import { useEffect, useMemo, useRef, useState } from 'react'

export function useWidth() {
  const ref = useRef(null)
  const [w, setW] = useState(600)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.floor(e.contentRect.width))))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

const niceTicks = (a, b, n = 5) => {
  if (!(b > a)) return [a]
  const span = b - a
  const step0 = span / n
  const mag = Math.pow(10, Math.floor(Math.log10(step0)))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag
  const out = []
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

const fmt = (v) => {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000 || a < 0.01) return v.toExponential(0)
  if (a >= 10) return v.toFixed(0)
  if (a >= 1) return v.toFixed(1)
  return v.toFixed(2)
}

/**
 * series: [{name, color, x: [], y: [], width?, dash?}]
 */
export function LineChart({
  series, height = 220, logY = false, xDomain, yDomain, band, playhead, onPickX, xLabel, yLabel, legend = true,
}) {
  const [ref, W] = useWidth()
  const [hover, setHover] = useState(null)
  const m = { l: 52, r: 12, t: 10, b: 34 }
  const iw = W - m.l - m.r
  const ih = height - m.t - m.b

  const dom = useMemo(() => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    for (const s of series) {
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i], y = s.y[i]
        if (!isFinite(y) || (logY && y <= 0)) continue
        if (xDomain && (x < xDomain[0] || x > xDomain[1])) continue
        x0 = Math.min(x0, x); x1 = Math.max(x1, x)
        y0 = Math.min(y0, y); y1 = Math.max(y1, y)
      }
    }
    if (xDomain) [x0, x1] = xDomain
    if (yDomain) [y0, y1] = yDomain
    if (!isFinite(y0)) { y0 = logY ? 1e-3 : 0; y1 = 1 }
    if (logY) { y0 = Math.max(y0, y1 * 1e-5) } else { y0 = Math.min(0, y0); y1 = y1 * 1.08 || 1 }
    return { x0, x1, y0, y1 }
  }, [series, xDomain, yDomain, logY])

  const sx = (x) => m.l + ((x - dom.x0) / (dom.x1 - dom.x0 || 1)) * iw
  const sy = logY
    ? (y) => m.t + ih - ((Math.log10(Math.max(y, dom.y0)) - Math.log10(dom.y0)) / (Math.log10(dom.y1) - Math.log10(dom.y0) || 1)) * ih
    : (y) => m.t + ih - ((y - dom.y0) / (dom.y1 - dom.y0 || 1)) * ih

  const paths = useMemo(() => series.map((s) => {
    let d = ''
    let pen = false
    for (let i = 0; i < s.x.length; i++) {
      const x = s.x[i], y = s.y[i]
      if (x < dom.x0 || x > dom.x1 || !isFinite(y) || (logY && y <= 0)) { pen = false; continue }
      d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`
      pen = true
    }
    return d
  }), [series, dom, W, height, logY]) // eslint-disable-line react-hooks/exhaustive-deps

  const xt = niceTicks(dom.x0, dom.x1, Math.max(3, Math.floor(iw / 70)))
  const yt = logY
    ? (() => { const a = Math.ceil(Math.log10(dom.y0)), b = Math.floor(Math.log10(dom.y1)); const o = []; for (let e = a; e <= b; e++) o.push(10 ** e); return o })()
    : niceTicks(dom.y0, dom.y1, 4)

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    if (px < m.l || px > m.l + iw) { setHover(null); return }
    const x = dom.x0 + ((px - m.l) / iw) * (dom.x1 - dom.x0)
    setHover(x)
  }
  const hv = hover == null ? null : series.map((s) => {
    let best = -1, bd = Infinity
    for (let i = 0; i < s.x.length; i++) { const d = Math.abs(s.x[i] - hover); if (d < bd) { bd = d; best = i } }
    return best < 0 ? null : { name: s.name, color: s.color, x: s.x[best], y: s.y[best] }
  }).filter(Boolean)

  return (
    <div ref={ref} className="chart">
      <svg width={W} height={height} onMouseMove={onMove} onMouseLeave={() => setHover(null)}
        onClick={(e) => { if (onPickX && hover != null) onPickX(hover); e.stopPropagation() }}
        style={{ cursor: onPickX ? 'crosshair' : 'default' }}>
        {band && (
          <rect x={sx(Math.max(band[0], dom.x0))} y={m.t} width={Math.max(0, sx(Math.min(band[1], dom.x1)) - sx(Math.max(band[0], dom.x0)))}
            height={ih} className="band" />
        )}
        {yt.map((v) => (
          <g key={`y${v}`}>
            <line x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} className="grid" />
            <text x={m.l - 6} y={sy(v) + 4} className="tick" textAnchor="end">{fmt(v)}</text>
          </g>
        ))}
        {xt.map((v) => (
          <g key={`x${v}`}>
            <line x1={sx(v)} x2={sx(v)} y1={m.t} y2={m.t + ih} className="grid" />
            <text x={sx(v)} y={m.t + ih + 16} className="tick" textAnchor="middle">{fmt(v)}</text>
          </g>
        ))}
        {paths.map((d, i) => (
          <path key={series[i].name} d={d} fill="none" stroke={series[i].color} strokeWidth={series[i].width || 1.5}
            strokeDasharray={series[i].dash} strokeLinejoin="round" opacity={0.95} />
        ))}
        {playhead != null && playhead >= dom.x0 && playhead <= dom.x1 && (
          <line x1={sx(playhead)} x2={sx(playhead)} y1={m.t} y2={m.t + ih} className="playhead" />
        )}
        {hover != null && <line x1={sx(hover)} x2={sx(hover)} y1={m.t} y2={m.t + ih} className="hoverline" />}
        {xLabel && <text x={m.l + iw / 2} y={height - 4} className="axis-label" textAnchor="middle">{xLabel}</text>}
        {yLabel && <text x={12} y={m.t + ih / 2} className="axis-label" textAnchor="middle" transform={`rotate(-90 12 ${m.t + ih / 2})`}>{yLabel}</text>}
      </svg>
      {hv && hv.length > 0 && (
        <div className="tooltip" style={{ left: Math.min(sx(hover) + 12, W - 170) }}>
          <div className="tt-x">{fmt(hv[0].x)} {xLabel?.match(/\((.*)\)/)?.[1] || ''}</div>
          {hv.map((h) => (
            <div key={h.name}><span className="dot" style={{ background: h.color }} />{h.name}: <b>{fmt(h.y)}</b></div>
          ))}
        </div>
      )}
      {legend && (
        <div className="legend">
          {series.map((s) => <span key={s.name}><span className="dot" style={{ background: s.color }} />{s.name}</span>)}
        </div>
      )}
    </div>
  )
}

/** Spectrogram heatmap (dB) drawn on canvas with a playhead. */
export function Spectrogram({ t, f, db, playhead, height = 200, fMax = 15, band, onPickX }) {
  const [ref, W] = useWidth()
  const cv = useRef(null)
  const m = { l: 52, r: 12, t: 8, b: 30 }
  const iw = W - m.l - m.r
  const ih = height - m.t - m.b
  const fIdx = useMemo(() => f.map((v, i) => [v, i]).filter(([v]) => v <= fMax).map(([, i]) => i), [f, fMax])

  useEffect(() => {
    const c = cv.current
    if (!c || !db?.length || !t.length) return
    const nx = t.length, ny = fIdx.length
    c.width = nx
    c.height = ny
    const ctx = c.getContext('2d')
    const img = ctx.createImageData(nx, ny)
    let lo = Infinity, hi = -Infinity
    for (const i of fIdx) for (let j = 0; j < nx; j++) { const v = db[i][j]; if (v < lo) lo = v; if (v > hi) hi = v }
    lo = Math.max(lo, hi - 40)
    for (let yi = 0; yi < ny; yi++) {
      const row = db[fIdx[ny - 1 - yi]]
      for (let x = 0; x < nx; x++) {
        const v = Math.min(1, Math.max(0, (row[x] - lo) / (hi - lo || 1)))
        const [r, g, b] = inferno(v)
        const k = (yi * nx + x) * 4
        img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [db, t, fIdx])

  const tMin = 0
  const tMax = t.length ? t[t.length - 1] + (t[1] - t[0] || 0) / 2 : 1
  const fm = f[fIdx[fIdx.length - 1]] || fMax
  const sx = (x) => m.l + ((x - tMin) / (tMax - tMin)) * iw
  const sy = (y) => m.t + ih - (y / fm) * ih
  return (
    <div ref={ref} className="chart">
      <div style={{ position: 'relative', height }}>
        <canvas ref={cv} style={{ position: 'absolute', left: m.l, top: m.t, width: iw, height: ih, imageRendering: 'auto' }} />
        <svg width={W} height={height} style={{ position: 'absolute', inset: 0, cursor: onPickX ? 'crosshair' : 'default' }}
          onClick={(e) => {
            if (!onPickX) return
            const r = e.currentTarget.getBoundingClientRect()
            const x = tMin + ((e.clientX - r.left - m.l) / iw) * (tMax - tMin)
            if (x >= tMin && x <= tMax) onPickX(x)
          }}>
          {niceTicks(0, fm, 5).map((v) => (
            <g key={v}><text x={m.l - 6} y={sy(v) + 4} className="tick" textAnchor="end">{v}</text></g>
          ))}
          {niceTicks(tMin, tMax, Math.max(3, Math.floor(iw / 70))).map((v) => (
            <text key={v} x={sx(v)} y={height - 12} className="tick" textAnchor="middle">{fmt(v)}</text>
          ))}
          {band && [band[0], band[1]].map((b) => <line key={b} x1={m.l} x2={m.l + iw} y1={sy(b)} y2={sy(b)} className="band-line" />)}
          {playhead != null && <line x1={sx(playhead)} x2={sx(playhead)} y1={m.t} y2={m.t + ih} className="playhead" />}
          <text x={12} y={m.t + ih / 2} className="axis-label" textAnchor="middle" transform={`rotate(-90 12 ${m.t + ih / 2})`}>Hz</text>
        </svg>
      </div>
    </div>
  )
}

/** Small XY (Lissajous) plots of band-passed fingertip displacement around the playhead. */
export function XYTraces({ trace, anchors, colors, playhead, window = 1.0, size = 120 }) {
  const { t, xy } = trace
  const dt = t.length > 1 ? t[1] - t[0] : 1 / 30
  const i1 = Math.max(0, Math.min(t.length - 1, Math.round((playhead || 0) / dt)))
  const i0 = Math.max(0, i1 - Math.round(window / dt))
  const lim = useMemo(() => {
    let mx = 0
    for (const a of anchors) for (const p of xy[a]) { mx = Math.max(mx, Math.abs(p[0]), Math.abs(p[1])) }
    return mx || 1
  }, [xy, anchors])
  const s = (v) => size / 2 + (v / lim) * (size / 2 - 6)
  return (
    <div className="xy-row">
      {anchors.map((a) => {
        const pts = xy[a].slice(i0, i1 + 1)
        const d = pts.map((p, i) => `${i ? 'L' : 'M'}${s(p[0]).toFixed(1)},${s(p[1]).toFixed(1)}`).join('')
        const last = pts[pts.length - 1]
        return (
          <div key={a} className="xy">
            <svg width={size} height={size}>
              <rect x={0.5} y={0.5} width={size - 1} height={size - 1} className="xy-frame" />
              <line x1={size / 2} x2={size / 2} y1={4} y2={size - 4} className="grid" />
              <line y1={size / 2} y2={size / 2} x1={4} x2={size - 4} className="grid" />
              <path d={d} fill="none" stroke={colors[a]} strokeWidth={1.4} opacity={0.9} />
              {last && <circle cx={s(last[0])} cy={s(last[1])} r={3} fill={colors[a]} />}
            </svg>
            <div className="xy-label" style={{ color: colors[a] }}>{a}</div>
          </div>
        )
      })}
      <div className="xy-note">±{lim.toFixed(2)} px<br />last {window}s</div>
    </div>
  )
}

// compact inferno approximation
const INF = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]]
function inferno(v) {
  const x = v * (INF.length - 1)
  const i = Math.min(INF.length - 2, Math.floor(x))
  const f = x - i
  const a = INF[i], b = INF[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}
