import { useWidth } from './Charts.jsx'
import { fmtTime } from '../api.js'

/** Whole-video strip: hand visibility, motion, chosen & suggested segment. Click to set segment start. */
const MK_COLORS = ['#f472b6', '#fbbf24', '#34d399', '#60a5fa', '#c084fc', '#fb923c', '#a3e635']

export default function Timeline({ scan, scanState, duration, segStart, segDur, recommended, playhead, onPick, markers }) {
  const [ref, W] = useWidth()
  const hasMk = markers && markers.length > 0
  const H = hasMk ? 74 : 54
  const mkColor = {}
  ;(markers || []).forEach((m) => { if (!(m.marker in mkColor)) mkColor[m.marker] = MK_COLORS[Object.keys(mkColor).length % MK_COLORS.length] })
  const dur = duration || scan?.duration || 1
  const sx = (t) => (t / dur) * W
  const segD = segDur > 0 ? segDur : dur - segStart

  let body = null
  if (scan) {
    const n = scan.t.length
    const bw = Math.max(1, W / Math.max(n, 1) + 0.5)
    const mv = scan.motion
    const mmax = Math.max(0.02, ...mv.map((v) => Math.min(v, 0.2)))
    const line = mv.map((v, i) => `${i ? 'L' : 'M'}${sx(scan.t[i]).toFixed(1)},${(50 - (Math.min(v, 0.2) / mmax) * 24).toFixed(1)}`).join('')
    body = (
      <>
        {scan.t.map((t, i) => (
          <rect key={i} x={sx(t)} y={4} width={bw} height={12}
            className={scan.good[i] ? (scan.full[i] ? 'tl-good' : 'tl-part') : scan.present[i] ? 'tl-part' : 'tl-none'} />
        ))}
        <path d={line} className="tl-motion" />
      </>
    )
  }

  return (
    <div className="timeline" ref={ref}>
      <div className="tl-head">
        <span>Video timeline</span>
        <span className="dim">
          <i className="sw good" /> hand visible <i className="sw part" /> partial <i className="sw none" /> absent
          <i className="sw motion" /> motion — click to set segment start
        </span>
        {scanState && scanState.state === 'running' && <span className="dim">Scanning… {Math.round((scanState.progress || 0) * 100)}%</span>}
      </div>
      <svg width={W} height={H} onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        onPick?.(Math.max(0, ((e.clientX - r.left) / W) * dur))
      }} style={{ cursor: 'pointer' }}>
        <rect x={0} y={0} width={W} height={H} className="tl-bg" />
        {body}
        {recommended && (
          <rect x={sx(recommended.start_s)} y={1} width={Math.max(2, sx(recommended.duration_s))} height={H - 2} className="tl-reco" />
        )}
        <rect x={sx(segStart)} y={1} width={Math.max(2, sx(segD))} height={H - 2} className="tl-seg" />
        {hasMk && markers.map((m, i) => (m.end_s != null ? (
          <rect key={i} x={sx(m.time_s)} y={H - 18} width={Math.max(2, sx(m.end_s) - sx(m.time_s))} height={8}
            fill={mkColor[m.marker]} opacity={0.55}><title>{`${m.marker}${m.value ? ` = ${m.value}` : ''}  ${m.time_s}-${m.end_s} s${m.notes ? `  (${m.notes})` : ''}`}</title></rect>
        ) : (
          <g key={i}>
            <line x1={sx(m.time_s)} x2={sx(m.time_s)} y1={H - 22} y2={H - 6} stroke={mkColor[m.marker]} strokeWidth={2} />
            <title>{`${m.marker}${m.value ? ` = ${m.value}` : ''}  at ${m.time_s} s${m.notes ? `  (${m.notes})` : ''}`}</title>
          </g>
        )))}
        {playhead != null && <line x1={sx(playhead)} x2={sx(playhead)} y1={0} y2={H} className="playhead" />}
        <text x={Math.min(sx(segStart) + 4, W - 90)} y={48} className="tl-label">{fmtTime(segStart)} +{segD.toFixed(0)}s</text>
      </svg>
      {hasMk && (
        <div className="tl-legend">
          {Object.entries(mkColor).map(([k, c]) => <span key={k}><i style={{ background: c }} />{k}</span>)}
        </div>
      )}
    </div>
  )
}
