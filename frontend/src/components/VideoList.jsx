import { useMemo, useState } from 'react'
import { api, fmtDur } from '../api.js'

const LEVEL_ORDER = { control: 0, low: 1, medium: 2, high: 3 }

function FpsBadge({ v }) {
  const cls = v.fps_status === 'ok' ? 'ok' : v.fps_status === 'low' ? 'warn' : 'bad'
  const title = v.fps_status === 'ok' ? 'Frame rate meets the 100 fps recommendation'
    : v.fps_status === 'low' ? `Below the recommended 100 fps: tremor band limited to ${v.max_band_hz} Hz`
      : 'Frame rate too low to analyse tremor'
  return <span className={`fps ${cls}`} title={title}>{v.fps.toFixed(0)} fps</span>
}

export default function VideoList({ videos, selected, onSelect, emptyText, onRemove }) {
  const [q, setQ] = useState('')
  const [level, setLevel] = useState('all')
  const levels = useMemo(() => {
    const s = new Set(videos.map((v) => v.meta?.level?.toLowerCase()).filter(Boolean))
    return ['all', ...Object.keys(LEVEL_ORDER).filter((l) => s.has(l))]
  }, [videos])
  const list = useMemo(() => videos.filter((v) => {
    if (q && !v.name.toLowerCase().includes(q.toLowerCase())) return false
    if (level !== 'all' && v.meta?.level?.toLowerCase() !== level) return false
    return true
  }).sort((a, b) => {
    const pa = a.meta?.pid ?? `~${a.name}`
    const pb = b.meta?.pid ?? `~${b.name}`
    if (pa !== pb) return pa.localeCompare(pb)
    if ((a.meta?.session ?? '') !== (b.meta?.session ?? '')) return (a.meta?.session ?? '').localeCompare(b.meta?.session ?? '')
    return (LEVEL_ORDER[a.meta?.level?.toLowerCase()] ?? 9) - (LEVEL_ORDER[b.meta?.level?.toLowerCase()] ?? 9)
  }), [videos, q, level])

  return (
    <div className="vlist">
      <div className="vlist-head">
        <input className="search" placeholder={`Search ${videos.length} videos…`} value={q} onChange={(e) => setQ(e.target.value)} />
        {levels.length > 2 && (
          <div className="seg small">
            {levels.map((l) => (
              <button key={l} className={`seg-btn ${level === l ? 'on' : ''}`} onClick={() => setLevel(l)}>{l}</button>
            ))}
          </div>
        )}
      </div>
      <div className="vlist-items">
        {list.map((v) => (
          <div key={v.name} className={`vitem ${selected === v.name ? 'on' : ''}`} role="button" tabIndex={0}
            onClick={() => onSelect(v.name)} onKeyDown={(e) => e.key === 'Enter' && onSelect(v.name)}>
            <img src={api.thumbUrl(v.name)} alt="" loading="lazy" />
            <div className="vmeta">
              <div className="vname" title={v.name}>
                {v.meta ? <>{v.meta.pid} · {v.meta.session}</> : v.stem}
              </div>
              <div className="vsub">
                {v.meta && <span className={`lvl ${v.meta.level.toLowerCase()}`}>{v.meta.level}</span>}
                <FpsBadge v={v} />
                {v.has_markers && <span className="mk" title="Marker CSV attached">markers</span>}
              </div>
              <div className="vsub dim">
                <span>{v.height >= 2160 ? '4K' : `${v.height}p`}</span>
                <span>{fmtDur(v.duration)}</span>
                {v.n_results > 0 && <span className="done" title="Processed variants">✓ {v.n_results}</span>}
              </div>
            </div>
            {onRemove && (
              <button className="vremove" title="Remove this video (deletes it and its results)"
                onClick={(e) => { e.stopPropagation(); onRemove(v.name) }}>×</button>
            )}
          </div>
        ))}
        {!list.length && <div className="dim pad">{emptyText || 'No videos.'}</div>}
      </div>
    </div>
  )
}
