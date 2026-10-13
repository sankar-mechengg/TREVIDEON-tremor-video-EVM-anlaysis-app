import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmtDur, url } from '../api.js'
import AnalysisPanel from '../components/AnalysisPanel.jsx'
import ComparePlayer from '../components/ComparePlayer.jsx'
import SettingsPanel from '../components/SettingsPanel.jsx'
import Timeline from '../components/Timeline.jsx'
import VideoList from '../components/VideoList.jsx'
import { Header } from '../site/Layout.jsx'
import { APP } from '../site/content.js'
import { LibraryBar, MarkersPanel, QueuePanel, Stepper, Tour, UploadPanel } from './Panels.jsx'

const LS_SETTINGS = 'trevideon.settings.v1'
const LS_SEL = 'trevideon.selected'
const LS_TOUR = 'trevideon.tour.seen'
const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d } catch { return d } }
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* ignore */ } }

const PRESETS = [
  { id: 'std', label: 'Standard', help: 'Recommended settings, whole camera view', values: { hand_isolation: false, work_height: 720, alpha: 15 } },
  { id: 'iso', label: 'Hand isolation', help: 'Segment the dominant hand, pin the wrist, then magnify', values: { hand_isolation: true } },
  { id: 'quick', label: 'Quick preview', help: '360p, 10 s: fast first look', values: { work_height: 360, duration_s: 10, hand_isolation: false } },
  { id: 'detail', label: 'High detail', help: '1080p working resolution (slower)', values: { work_height: 1080 } },
]

const TOUR = [
  { target: 'tour-source', title: '1 · Add your videos', text: 'Local: choose the folder that holds your videos with “Browse…”. Online: drop MP4 files (and optional marker CSVs). Then pick a video in the list; the badge shows whether it meets the 100 fps recommendation.' },
  { target: 'tour-player', title: '2 · Look at the video', text: 'The left pane plays the raw video, the right one the processed result. Wheel = zoom, drag = pan, space = play, ←/→ = single frames.' },
  { target: 'tour-timeline', title: '3 · Choose the segment', text: 'A quick scan shows where the hand is visible (green) and suggests the steadiest segment (dashed). Click the timeline to move the segment; markers from your CSV appear underneath.' },
  { target: 'tour-settings', title: '4 · Set the parameters', text: 'Start with a preset. “Hand isolation” removes the background and pins the wrist so voluntary movement disappears. Every setting has a “?” with an explanation.' },
  { target: 'tour-analyse', title: '5 · Analyse', text: 'Analyse the selected video, or “Run all” to queue every video. Progress is shown in the queue; you can keep working meanwhile.' },
  { target: 'tour-results', title: '6 · Read the results', text: 'Frequency, amplitude, spectra, envelope and per-finger metrics appear below the player. Download everything as one ZIP.' },
]

export default function Analyzer() {
  const [cfg, setCfg] = useState(null)
  const [library, setLibrary] = useState(null)
  const [session, setSession] = useState(null)
  const [sessionInfo, setSessionInfo] = useState(null)
  const [videos, setVideos] = useState([])
  const [sel, setSel] = useState(() => load(LS_SEL, null))
  const [settingsMap, setSettingsMap] = useState(() => load(LS_SETTINGS, {}))
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [jobs, setJobs] = useState([])
  const [history, setHistory] = useState([])
  const [scanState, setScanState] = useState(null)
  const [markers, setMarkers] = useState(null)
  const [time, setTime] = useState(0)
  const [seekReq, setSeekReq] = useState(null)
  const [rightMode, setRightMode] = useState('overlay')
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [tour, setTour] = useState(false)
  const [visited, setVisited] = useState({})
  const [selectedVariant, setSelectedVariant] = useState('')
  const [fpsWarningDismissed, setFpsWarningDismissed] = useState(false)
  const lastT = useRef(0)
  const rawTime = useRef(0)
  const hosted = cfg?.mode === 'hosted'

  const video = videos.find((v) => v.name === sel)
  const settings = useMemo(() => (cfg ? { ...cfg.defaults, ...(settingsMap[sel]?.values || {}) } : null), [cfg, settingsMap, sel])

  const setSettings = useCallback((fn) => {
    setSettingsMap((m) => {
      const cur = { ...cfg.defaults, ...(m[sel]?.values || {}) }
      const next = typeof fn === 'function' ? fn(cur) : fn
      const out = { ...m, [sel]: { values: next, touched: true } }
      save(LS_SETTINGS, out)
      return out
    })
    setVisited((v) => ({ ...v, settings: true }))
  }, [cfg, sel])

  const refreshVideos = useCallback(() => api.videos().then((v) => {
    setVideos(v)
    setSel((s) => (s && v.some((x) => x.name === s) ? s : v[0]?.name ?? null))
  }).catch((e) => setError(e.message)), [])

  // ---- start-up: config, session (hosted) or library (local), videos
  useEffect(() => {
    (async () => {
      try {
        const c = await api.config()
        setCfg(c)
        if (c.mode === 'hosted') {
          setSession(await api.ensureSession(c.mode))
        } else {
          setLibrary(await api.library())
        }
        await refreshVideos()
        if (!load(LS_TOUR, false)) { setTour(true); save(LS_TOUR, true) }
      } catch (e) {
        setError(`Cannot reach the ${APP.name} server: ${e.message}`)
      }
    })()
  }, [refreshVideos])

  // hosted: session countdown + best-effort expiry when the page is closed
  useEffect(() => {
    if (!hosted) return
    const tick = () => api.sessionInfo().then(setSessionInfo).catch(() => {})
    tick()
    const h = setInterval(tick, 30000)
    const bye = () => api.endSessionBeacon()
    window.addEventListener('pagehide', bye)
    return () => { clearInterval(h); window.removeEventListener('pagehide', bye) }
  }, [hosted, session])

  useEffect(() => { if (sel) save(LS_SEL, sel) }, [sel])

  // ---- per-video: reset view, scan, history, markers
  const refreshMarkers = useCallback(() => {
    if (sel) api.markers(sel).then(setMarkers).catch(() => setMarkers(null))
  }, [sel])

  useEffect(() => {
    if (!sel) return
    setResult(null); setStatus(null); setScanState(null); setTime(0); setMarkers(null)
    setSelectedVariant(''); setFpsWarningDismissed(false)
    api.results(sel).then(setHistory).catch(() => setHistory([]))
    refreshMarkers()
    let stop = false
    let timer
    const poll = () => {
      api.scan(sel).then((s) => {
        if (stop) return
        setScanState(s)
        if (s.state === 'running') timer = setTimeout(poll, 900)
      }).catch(() => {})
    }
    poll()
    return () => { stop = true; clearTimeout(timer) }
  }, [sel, refreshMarkers])

  const recommended = scanState?.state === 'done' ? scanState.recommended : null

  useEffect(() => {
    if (!recommended || !cfg || !sel) return
    setSettingsMap((m) => {
      if (m[sel]?.touched || m[sel]?.auto) return m
      const out = { ...m, [sel]: { values: { ...cfg.defaults, ...(m[sel]?.values || {}), start_s: recommended.start_s }, auto: true } }
      save(LS_SETTINGS, out)
      return out
    })
  }, [recommended, cfg, sel])

  // ---- cache status (debounced)
  useEffect(() => {
    if (!sel || !settings) return
    const h = setTimeout(() => {
      api.check(sel, settings).then((s) => { setStatus(s); setNotice(s.notes?.length ? s.notes.join(' · ') : null) }).catch(() => setStatus(null))
    }, 250)
    return () => clearTimeout(h)
  }, [sel, settings])

  const loadResult = useCallback((ka, kb, name = sel) => {
    api.result(name, ka, kb).then((r) => { setResult({ ...r, video: name }); setError(null) }).catch((e) => setError(e.message))
  }, [sel])

  useEffect(() => {
    if (status?.stage_a_cached && status?.stage_b_cached &&
        (result?.key_a !== status.key_a || result?.key_b !== status.key_b || result?.video !== sel)) {
      loadResult(status.key_a, status.key_b)
    }
  }, [status, result, loadResult, sel])

  // ---- job polling (all jobs of this user / session)
  const active = jobs.some((j) => ['queued', 'running'].includes(j.state))
  const doneSeen = useRef(new Set())
  useEffect(() => {
    let stop = false
    let timer
    const poll = () => {
      api.jobs().then((js) => {
        if (stop) return
        setJobs(js)
        js.filter((j) => j.state === 'done' && !doneSeen.current.has(j.id)).forEach((j) => {
          doneSeen.current.add(j.id)
          if (j.video === sel) {
            api.results(sel).then(setHistory)
            api.check(sel, settings).then(setStatus)
          }
          refreshVideos()
        })
        if (js.some((j) => ['queued', 'running'].includes(j.state))) timer = setTimeout(poll, 1000)
      }).catch(() => {})
    }
    poll()
    return () => { stop = true; clearTimeout(timer) }
  }, [active, sel, settings, refreshVideos])

  const onProcess = async () => {
    try {
      setVisited((v) => ({ ...v, settings: true }))
      const r = await api.process(sel, settings)
      if (r.cached) loadResult(r.key_a, r.key_b)
      else setJobs((js) => [r.job, ...js.filter((j) => j.id !== r.job.id)])
      if (r.notes?.length) setNotice(r.notes.join(' · '))
    } catch (e) { setError(e.message) }
  }

  const onRunAll = async () => {
    const items = videos.filter((v) => v.fps_status !== 'blocked').map((v) => {
      const own = settingsMap[v.name]
      return { video: v.name, settings: own?.touched || own?.auto ? { ...cfg.defaults, ...own.values } : { ...settings, start_s: undefined, auto_start: true } }
    })
    if (!window.confirm(`Queue ${items.length} videos for analysis with the current settings?\nVideos without a chosen segment use the steadiest segment found by the scan. This can take a while; you can keep working.`)) return
    try {
      const r = await api.processAll(items)
      const errs = r.filter((x) => x.error)
      setNotice(`${r.length - errs.length} videos queued${errs.length ? ` · ${errs.length} skipped: ${errs.map((e) => e.video).join(', ')}` : ''}`)
      api.jobs().then(setJobs)
    } catch (e) { setError(e.message) }
  }

  const onDeleteAll = async () => {
    if (!window.confirm('Delete all your uploaded videos and results from the server now?')) return
    await api.deleteSession().catch(() => {})
    setVideos([]); setSel(null); setResult(null); setHistory([]); setJobs([])
    setSession(await api.ensureSession('hosted'))
    setNotice('All your data was deleted from the server. A new, empty session has started.')
  }

  const onRemoveVideo = async (name) => {
    if (!window.confirm(`Remove ${name} and its results from the server?`)) return
    await api.deleteVideo(name).catch((e) => setError(e.message))
    refreshVideos()
  }

  const onTime = useCallback((t) => {
    rawTime.current = t
    if (Math.abs(t - lastT.current) > 0.03) { lastT.current = t; setTime(t) }
  }, [])

  const onReset = () => setSettings({ ...cfg.defaults, ...(recommended ? { start_s: recommended.start_s } : {}) })
  const onPreset = (p) => setSettings((s) => ({ ...s, ...p.values }))
  const applyVariant = (r, ov) => setSettings({ ...cfg.defaults, ...r.params, ...(ov?.overlay || {}) })

  const scrollTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'center' })

  if (error && !cfg) {
    return (
      <div className="app-shell">
        <Header compact />
        <div className="fatal">
          <div>
            <h2>{error}</h2>
            <p className="dim">Run <code>.\run.ps1</code> in the TREVIDEON-app folder to start the local server, then reload this page.</p>
            <Link className="btn" to="/">Back to the home page</Link>
          </div>
        </div>
      </div>
    )
  }
  if (!cfg) return <div className="app-shell"><Header compact /><div className="fatal">Loading…</div></div>

  const showing = result && result.analysis.video === sel && result.video === sel
  const busyHere = jobs.find((j) => j.video === sel && ['queued', 'running'].includes(j.state))
  const leftSrc = showing ? url(result.urls.original) : sel ? api.rawUrl(sel) : null
  const rightUrl = showing ? (rightMode === 'overlay' && result.urls.overlay ? result.urls.overlay
    : rightMode === 'isolated' && result.urls.isolated ? result.urls.isolated : result.urls.magnified) : null
  const segStart = showing ? result.analysis.start_s : settings?.start_s ?? 0
  const stale = showing && status && (result.key_a !== status.key_a || result.key_b !== status.key_b)
  const iso = !!settings?.hand_isolation

  const steps = [
    { id: 'videos', label: 'Videos', target: 'tour-source', state: video ? 'done' : 'now' },
    { id: 'segment', label: 'Segment', target: 'tour-timeline', state: !video ? 'todo' : (settingsMap[sel]?.touched || recommended) ? 'done' : 'now' },
    { id: 'settings', label: 'Parameters', target: 'tour-settings', state: !video ? 'todo' : visited.settings || showing ? 'done' : 'now' },
    { id: 'analyse', label: 'Analyse', target: 'tour-analyse', state: showing ? 'done' : busyHere ? 'now' : 'todo' },
    { id: 'results', label: 'Results', target: 'tour-results', state: showing ? 'done' : 'todo' },
  ]

  return (
    <div className="app-shell">
      <Header compact />
      <Stepper steps={steps} onStep={(s) => scrollTo(s.target)} onTour={() => setTour(true)}
        right={hosted ? (
          <div className="session-box">
            <span className="dim" title="Your files are deleted automatically after this much inactivity">
              🔒 auto-delete in {sessionInfo?.expires_in_s != null ? Math.ceil(sessionInfo.expires_in_s / 60) : cfg.ttl_min} min
            </span>
            <button className="btn sm danger" onClick={onDeleteAll}>Delete my data</button>
          </div>
        ) : (
          <span className="dim mode-pill" title="Running on this computer: nothing leaves it">💻 Local · {cfg.compute?.gpu ? 'GPU' : 'CPU'}</span>
        )} />

      <div className="main">
        <aside className="left" id="tour-source">
          {hosted
            ? <UploadPanel limits={cfg.limits} onUploaded={(v) => { refreshVideos().then(() => setSel(v)) }} onError={setError} />
            : <LibraryBar library={library} onChanged={(lib) => { setLibrary(lib); setSel(null); refreshVideos() }} onError={setError} />}
          <VideoList videos={videos} selected={sel} onSelect={setSel} onRemove={hosted ? onRemoveVideo : null}
            emptyText={hosted ? 'Drop videos above to start.' : 'No videos in this folder. Use “Browse…” to choose another one.'} />
        </aside>

        <section className="center">
          {error && <div className="banner err">{error} <button className="btn ghost sm" onClick={() => setError(null)}>×</button></div>}
          {notice && <div className="banner info">{notice} <button className="btn ghost sm" onClick={() => setNotice(null)}>×</button></div>}
          {video && video.fps_status !== 'ok' && !fpsWarningDismissed && (
            <div className={`banner ${video.fps_status === 'blocked' ? 'err' : 'warn'}`}>
              <span>
                {video.fps_status === 'blocked'
                  ? `${video.fps.toFixed(0)} fps is too low for tremor analysis (minimum ${cfg.limits.min_fps} fps).`
                  : `${video.fps.toFixed(0)} fps: below the recommended ${cfg.limits.recommended_fps} fps. Only frequencies up to ${video.max_band_hz} Hz can be analysed and fast motion is blurred.`}
                {' '}<Link to="/guide#fps">Why?</Link>
              </span>
              <button className="btn ghost sm" onClick={() => setFpsWarningDismissed(true)} aria-label="Close frame-rate warning">×</button>
            </div>
          )}
          {stale && <div className="banner warn">Showing a previous result: settings changed{status.stage_a_cached ? ' (overlay only)' : ''}. Press “Analyse this video” to update.</div>}

          {video && (
            <div className="videobar">
              <div className="vb-title">
                <b>{video.meta ? `${video.meta.pid} · ${video.meta.session} · ${video.meta.level}` : video.stem}</b>
                <span className="dim">{video.width}×{video.height} · {video.fps.toFixed(0)} fps · {fmtDur(video.duration)} · {(video.size_bytes / 1e6).toFixed(0)} MB</span>
              </div>
              <div className="seg" title="Switch between the standard analysis and the hand-isolated (wrist-anchored) analysis">
                <button className={`seg-btn ${!iso ? 'on' : ''}`} onClick={() => setSettings((s) => ({ ...s, hand_isolation: false }))}>Standard</button>
                <button className={`seg-btn ${iso ? 'on' : ''}`} onClick={() => setSettings((s) => ({ ...s, hand_isolation: true }))}>Hand isolation</button>
              </div>
              {history.length > 0 && (
                <select className="variants" value={selectedVariant} onChange={(e) => {
                  const value = e.target.value
                  setSelectedVariant(value)
                  const [i, j] = value.split(':').map(Number)
                  applyVariant(history[i], history[i].overlays[j])
                }}>
                  <option value="" disabled>Earlier results ({history.length})</option>
                  {history.map((r, i) => r.overlays.map((o, j) => (
                    <option key={`${i}:${j}`} value={`${i}:${j}`}>
                      {r.created.slice(5, 16)} · {r.params.hand_isolation ? 'isolated' : 'standard'} · {r.params.start_s}s+{r.params.duration_s}s
                      {' '}· {r.params.freq_lo}–{r.params.freq_hi} Hz{r.params.repair_duplicates ? ' · repaired' : ''}
                    </option>
                  )))}
                </select>
              )}
            </div>
          )}

          <div id="tour-player">
            <ComparePlayer
              left={leftSrc ? { src: leftSrc, label: showing ? `${iso ? 'Wrist-anchored hand' : 'Original'} · ${result.analysis.start_s.toFixed(1)}–${(result.analysis.start_s + result.analysis.duration_s).toFixed(1)} s` : 'Raw video (preview)' } : null}
              right={{ src: rightUrl ? url(rightUrl) : null, label: showing ? (rightMode === 'overlay' ? 'Tremor-envelope overlay' : rightMode === 'isolated' ? 'Isolated input' : 'Magnified (EVM)') : 'Result' }}
              fps={video?.fps || 30}
              onTime={onTime}
              seekRequest={seekReq}
              initialTime={showing ? 0 : settings?.start_s}
              leftExtra={!showing && sel && (
                <button className="btn sm" title="Use the current preview time as the segment start"
                  onClick={() => setSettings((s) => ({ ...s, start_s: Math.round(rawTime.current * 2) / 2 }))}>Set start here</button>
              )}
              rightTabs={showing && (
                <div className="seg small">
                  <button className={`seg-btn ${rightMode === 'overlay' ? 'on' : ''}`} onClick={() => setRightMode('overlay')}>Overlay</button>
                  <button className={`seg-btn ${rightMode === 'magnified' ? 'on' : ''}`} onClick={() => setRightMode('magnified')}>Magnified</button>
                  {result.urls.isolated && <button className={`seg-btn ${rightMode === 'isolated' ? 'on' : ''}`} onClick={() => setRightMode('isolated')}>Isolated</button>}
                </div>
              )}
              rightPlaceholder={busyHere ? (
                <div className="ph-job">
                  <div className="spinner" />
                  <div>{busyHere.state === 'queued' ? `Waiting in queue (#${busyHere.queue_position})` : busyHere.message}</div>
                  <div className="bar"><div style={{ width: `${(busyHere.progress || 0) * 100}%` }} /></div>
                  <div className="dim mono">{Math.round((busyHere.progress || 0) * 100)}%{busyHere.eta_s != null ? ` · ETA ${Math.round(busyHere.eta_s)}s` : ''}</div>
                </div>
              ) : (
                <div className="ph-idle">
                  <div className="big">▶</div>
                  <div>{sel ? 'Not analysed with these settings yet.' : 'Choose a video on the left.'}</div>
                  {sel && <div className="dim">Pick a segment on the timeline, then press <b>Analyse this video</b>.</div>}
                </div>
              )}
            />
          </div>

          <div id="tour-timeline">
            <Timeline scan={scanState?.scan} scanState={scanState} duration={video?.duration} segStart={segStart}
              segDur={showing ? result.analysis.duration_s : settings?.duration_s} recommended={recommended}
              playhead={showing ? result.analysis.start_s + time : time} markers={markers?.rows}
              onPick={(t) => {
                setSettings((s) => ({ ...s, start_s: Math.round(t * 2) / 2 }))
                if (!showing) setSeekReq({ t })
              }} />
          </div>

          <MarkersPanel video={sel} markers={markers} onChange={() => { refreshMarkers(); refreshVideos(); if (status) loadResult(status.key_a, status.key_b) }} />

          <div id="tour-results">
            {showing ? (
              <>
                <div className="result-actions">
                  <span className="dim">Result {result.key_a} · {result.analysis.mode === 'hand_isolation' ? 'hand isolation' : 'standard'} · {result.analysis.device === 'gpu' ? 'GPU' : 'CPU'} · {result.analysis.processing_s}s</span>
                  <a className="btn sm primary" href={url(result.urls.bundle)} download>Download all (ZIP)</a>
                  <a className="btn sm" href={url(`${result.urls.bundle}&videos=false`)} download>Metrics only</a>
                </div>
                <AnalysisPanel analysis={result.analysis} overlay={result.overlay} time={time} onSeek={(t) => setSeekReq({ t })}
                  markerMetrics={result.marker_metrics} />
              </>
            ) : (
              <div className="empty-analysis dim">Tremor frequency, amplitude, spectra, envelopes and fingertip trajectories appear here after the analysis.</div>
            )}
          </div>
        </section>

        <div className="right" id="tour-settings">
          <SettingsPanel schema={cfg.schema} settings={settings} setSettings={setSettings} onReset={onReset}
            status={status} onProcess={onProcess} job={busyHere} videoInfo={video} gpu={cfg.compute?.gpu}
            onCancel={() => busyHere && api.cancel(busyHere.id).then(() => api.jobs().then(setJobs))}
            recommended={recommended} presets={PRESETS} onPreset={onPreset}
            onAutoSegment={() => recommended && setSettings((s) => ({ ...s, start_s: recommended.start_s }))}
            onRunAll={onRunAll} nVideos={videos.length} />
          <QueuePanel jobs={jobs} onCancel={(j) => api.cancel(j.id).then(() => api.jobs().then(setJobs))}
            onOpen={(j) => setSel(j.video)} />
        </div>
      </div>
      <Tour steps={TOUR} open={tour} onClose={() => setTour(false)} />
    </div>
  )
}
