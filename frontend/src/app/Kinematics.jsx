import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmtDur, url } from '../api.js'
import ComparePlayer from '../components/ComparePlayer.jsx'
import Hand3DViewer, { autoAlpha, useViewerControls } from '../components/Hand3DViewer.jsx'
import KinPlots from '../components/KinPlots.jsx'
import SettingsPanel from '../components/SettingsPanel.jsx'
import Timeline from '../components/Timeline.jsx'
import VideoList from '../components/VideoList.jsx'
import { Header } from '../site/Layout.jsx'
import { APP } from '../site/content.js'
import { LibraryBar, ModeTabs, QueuePanel, Stepper, Tour, UploadPanel } from './Panels.jsx'

const LS_SETTINGS = 'trevideon.k3d.settings.v1'
const LS_EVM = 'trevideon.settings.v1'
const LS_SEL = 'trevideon.selected'
const LS_TOUR = 'trevideon.k3d.tour.seen'
const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d } catch { return d } }
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* ignore */ } }

const PRESETS = [
  { id: 'std', label: 'Recommended', help: 'WiLoR (or MediaPipe), head motion removed, noise floor measured', values: { model: 'auto', world_frame: 'stabilised', noise_floor: true } },
  { id: 'quick', label: 'Quick look', help: '10 s, no noise-floor re-runs', values: { duration_s: 10, noise_floor: false } },
  { id: 'mp', label: 'MediaPipe only', help: 'CPU model (no MANO / GPU needed)', values: { model: 'mediapipe' } },
  { id: 'cam', label: 'Static camera', help: 'Tripod / desk camera: no head-motion removal', values: { world_frame: 'camera', view: 'third_person' } },
]

const TOUR = [
  { target: 'tour-source', title: '1 · Choose a video', text: 'The same video list as the EVM tab. Pick a video; the segment you analysed in the EVM tab can be reused with one click.' },
  { target: 'tour-player', title: '2 · Video and 3D hand', text: 'Left: the segmented hand (original camera geometry). Right: the reconstructed 3D hand with the world, wrist and finger coordinate frames. Drag to orbit, wheel to zoom.' },
  { target: 'tour-viewer', title: '3 · Show the tremor', text: '“Frozen” wrist removes the hand’s overall motion; the magnify slider multiplies only the finger vibration (tremor band). Trails and ellipsoids show its paths and principal directions.' },
  { target: 'tour-settings', title: '4 · Parameters', text: 'Model, coordinate frames, tremor band and noise floor. Every setting has a “?”.' },
  { target: 'tour-results', title: '5 · Results', text: 'Fingertip motion relative to the wrist, orientation, velocity and acceleration, spectra with noise floors, spectrograms, coherence and per-finger amplitudes, synced with the video. Download CSV, MATLAB, BVH, glTF and figures.' },
]

function estimate3d(s, v, gpu) {
  if (!v) return null
  const d = Math.min(s.duration_s, Math.max(0, v.duration - s.start_s))
  const perFrame = gpu ? 0.18 + 0.2 + (s.noise_floor ? 0.04 : 0) : 0.16 + 0.15
  return Math.max(1, Math.round((d * v.fps * perFrame * (v.height / 1080) ** 0.5) / 60))
}

export default function Kinematics() {
  const [cfg, setCfg] = useState(null)
  const [library, setLibrary] = useState(null)
  const [session, setSession] = useState(null)
  const [videos, setVideos] = useState([])
  const [sel, setSel] = useState(() => load(LS_SEL, null))
  const [settingsMap, setSettingsMap] = useState(() => load(LS_SETTINGS, {}))
  const [status, setStatus] = useState(null)
  const [meta, setMeta] = useState(null)          // /api/k3d/{name}/{key}
  const [data, setData] = useState(null)          // kinematics.json
  const [loading, setLoading] = useState(false)
  const [jobs, setJobs] = useState([])
  const [history, setHistory] = useState([])
  const [scanState, setScanState] = useState(null)
  const [markers, setMarkers] = useState(null)
  const [time, setTime] = useState(0)
  const [seekReq, setSeekReq] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [tour, setTour] = useState(false)
  const [viewerInfo, setViewerInfo] = useState({})
  const [ctl, ctlApi, setCtl] = useViewerControls()
  const lastT = useRef(0)
  const rawTime = useRef(0)
  const hosted = cfg?.mode === 'hosted'
  const K3 = cfg?.kin3d

  const video = videos.find((v) => v.name === sel)
  const settings = useMemo(() => (K3 ? { ...K3.defaults, ...(settingsMap[sel]?.values || {}) } : null), [K3, settingsMap, sel])
  const setSettings = useCallback((fn) => {
    setSettingsMap((m) => {
      const cur = { ...K3.defaults, ...(m[sel]?.values || {}) }
      const next = typeof fn === 'function' ? fn(cur) : fn
      const out = { ...m, [sel]: { values: next, touched: true } }
      save(LS_SETTINGS, out)
      return out
    })
  }, [K3, sel])

  const refreshVideos = useCallback(() => api.videos().then((v) => {
    setVideos(v)
    setSel((s) => (s && v.some((x) => x.name === s) ? s : v[0]?.name ?? null))
  }).catch((e) => setError(e.message)), [])

  useEffect(() => {
    (async () => {
      try {
        const c = await api.config()
        setCfg(c)
        if (c.mode === 'hosted') setSession(await api.ensureSession(c.mode))
        else setLibrary(await api.library())
        await refreshVideos()
        if (!load(LS_TOUR, false)) { setTour(true); save(LS_TOUR, true) }
      } catch (e) { setError(`Cannot reach the ${APP.name} server: ${e.message}`) }
    })()
  }, [refreshVideos])
  useEffect(() => { if (sel) save(LS_SEL, sel) }, [sel])

  // per video
  useEffect(() => {
    if (!sel) return
    setMeta(null); setData(null); setStatus(null); setScanState(null); setTime(0); setMarkers(null)
    api.k3dResults(sel).then(setHistory).catch(() => setHistory([]))
    api.markers(sel).then(setMarkers).catch(() => setMarkers(null))
    let stop = false
    let timer
    const poll = () => api.scan(sel).then((s) => {
      if (stop) return
      setScanState(s)
      if (s.state === 'running') timer = setTimeout(poll, 900)
    }).catch(() => {})
    poll()
    return () => { stop = true; clearTimeout(timer) }
  }, [sel])

  const recommended = scanState?.state === 'done' ? scanState.recommended : null
  const evmSeg = useMemo(() => {
    const e = load(LS_EVM, {})[sel]?.values
    return e && e.start_s != null ? { start_s: e.start_s, duration_s: e.duration_s || 20 } : null
  }, [sel])

  useEffect(() => {
    if (!K3 || !sel) return
    setSettingsMap((m) => {
      if (m[sel]?.touched || m[sel]?.auto) return m
      const start = evmSeg?.start_s ?? recommended?.start_s
      if (start == null) return m
      const out = { ...m, [sel]: { values: { ...K3.defaults, ...(m[sel]?.values || {}), start_s: start, ...(evmSeg ? { duration_s: evmSeg.duration_s } : {}) }, auto: true } }
      save(LS_SETTINGS, out)
      return out
    })
  }, [recommended, evmSeg, K3, sel])

  // cache status
  useEffect(() => {
    if (!sel || !settings) return
    const h = setTimeout(() => {
      api.k3dCheck(sel, settings).then((s) => { setStatus({ ...s, video: sel }); if (s.notes?.length) setNotice(s.notes.join(' · ')) }).catch(() => setStatus(null))
    }, 250)
    return () => clearTimeout(h)
  }, [sel, settings])

  const loadResult = useCallback(async (k, kv, name = sel) => {
    try {
      setLoading(true)
      const m = await api.k3dResult(name, k, kv)
      const d = await api.fetchJson(m.urls.data)
      setMeta({ ...m, video: name }); setData(d); setError(null)
      setSeekReq({ t: 0 })
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }, [sel])

  useEffect(() => {
    if (status?.video === sel && status?.k_cached && status?.v_cached && (meta?.key !== status.key || meta?.key_v !== status.key_v || meta?.video !== sel) && !loading) {
      loadResult(status.key, status.key_v)
    }
  }, [status, meta, loadResult, sel, loading])

  // jobs
  const active = jobs.some((j) => ['queued', 'running'].includes(j.state))
  const doneSeen = useRef(new Set())
  useEffect(() => {
    let stop = false
    let timer
    const poll = () => api.jobs().then((js) => {
      if (stop) return
      setJobs(js)
      js.filter((j) => j.state === 'done' && !doneSeen.current.has(j.id)).forEach((j) => {
        doneSeen.current.add(j.id)
        if (j.video === sel && settings) {
          api.k3dResults(sel).then(setHistory).catch(() => {})
          api.k3dCheck(sel, settings).then((st) => setStatus({ ...st, video: sel })).catch(() => {})
        }
        refreshVideos()
      })
      if (js.some((j) => ['queued', 'running'].includes(j.state))) timer = setTimeout(poll, 1000)
    }).catch(() => {})
    poll()
    return () => { stop = true; clearTimeout(timer) }
  }, [active, sel, settings, refreshVideos])

  const onProcess = async () => {
    try {
      const r = await api.k3dProcess(sel, settings)
      if (r.cached) loadResult(r.key, r.key_v)
      else setJobs((js) => [r.job, ...js.filter((j) => j.id !== r.job.id)])
      if (r.notes?.length) setNotice(r.notes.join(' · '))
    } catch (e) { setError(e.message) }
  }
  const onRunAll = async () => {
    const items = videos.filter((v) => v.fps_status !== 'blocked').map((v) => {
      const own = settingsMap[v.name]
      const evm = load(LS_EVM, {})[v.name]?.values
      const seg = own?.touched || own?.auto ? own.values : evm?.start_s != null ? { start_s: evm.start_s, duration_s: evm.duration_s } : {}
      return { video: v.name, settings: { ...settings, ...seg } }
    })
    if (!window.confirm(`Queue ${items.length} videos for 3D kinematics?\nEach uses its own segment (3D tab, else the EVM tab's segment, else ${settings.start_s}s). Each 20 s segment takes a few minutes on a GPU.`)) return
    try {
      const r = await api.k3dProcessAll(items)
      const errs = r.filter((x) => x.error)
      setNotice(`${r.length - errs.length} videos queued${errs.length ? ` · ${errs.length} skipped` : ''}`)
      api.jobs().then(setJobs)
    } catch (e) { setError(e.message) }
  }
  const onTime = useCallback((t) => {
    rawTime.current = t
    if (Math.abs(t - lastT.current) > 0.03) { lastT.current = t; setTime(t) }
  }, [])
  const scrollTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'center' })

  if (error && !cfg) {
    return (
      <div className="app-shell"><Header compact />
        <div className="fatal"><div><h2>{error}</h2>
          <p className="dim">Run <code>.\run.ps1</code> in the TREVIDEON-app folder to start the local server, then reload this page.</p>
          <Link className="btn" to="/">Back to the home page</Link></div></div>
      </div>
    )
  }
  if (!cfg || !K3) return <div className="app-shell"><Header compact /><div className="fatal">Loading…</div></div>

  const showing = data && meta?.video === sel && data.video === sel
  const busyHere = jobs.find((j) => j.video === sel && j.kind === 'kin3d' && ['queued', 'running'].includes(j.state))
  const leftSrc = showing && meta.urls.display ? url(meta.urls.display) : sel ? api.rawUrl(sel) : null
  const segStart = showing ? data.start_s : settings?.start_s ?? 0
  const stale = showing && status && (meta.key !== status.key || meta.key_v !== status.key_v)
  const wilor = K3.models?.wilor
  const steps = [
    { id: 'videos', label: 'Video', target: 'tour-source', state: video ? 'done' : 'now' },
    { id: 'segment', label: 'Segment', target: 'tour-timeline', state: !video ? 'todo' : settingsMap[sel] ? 'done' : 'now' },
    { id: 'settings', label: 'Parameters', target: 'tour-settings', state: !video ? 'todo' : showing ? 'done' : 'now' },
    { id: 'analyse', label: 'Analyse', target: 'tour-settings', state: showing ? 'done' : busyHere ? 'now' : 'todo' },
    { id: 'results', label: 'Results', target: 'tour-results', state: showing ? 'done' : 'todo' },
  ]

  const viewer = showing ? (
    <Hand3DViewer data={data} meshUrl={meta.urls.mesh} time={time} controls={ctl}
      hasVibeSkin={!!K3.hand_model?.vibeskin} onInfo={(i) => setViewerInfo((v) => ({ ...v, ...i }))} />
  ) : null

  return (
    <div className="app-shell">
      <Header compact />
      <Stepper steps={steps} onStep={(s) => scrollTo(s.target)} onTour={() => setTour(true)} left={<ModeTabs current="3d" />}
        right={<span className="dim mode-pill" title="3D model available on this server">
          {wilor?.ready ? (wilor.remote ? '🧠 WiLoR (GPU worker)' : '🧠 WiLoR + MANO') : '🧠 MediaPipe 3D'}{hosted ? '' : ` · ${cfg.compute?.gpu ? 'GPU' : 'CPU'}`}</span>} />

      <div className="main">
        <aside className="left" id="tour-source">
          {hosted
            ? <UploadPanel limits={cfg.limits} onUploaded={(v) => { refreshVideos().then(() => setSel(v)) }} onError={setError} />
            : <LibraryBar library={library} onChanged={(lib) => { setLibrary(lib); setSel(null); refreshVideos() }} onError={setError} />}
          <VideoList videos={videos} selected={sel} onSelect={setSel}
            emptyText={hosted ? 'Drop videos above to start.' : 'No videos in this folder. Use “Browse…” to choose another one.'} />
        </aside>

        <section className="center">
          {error && <div className="banner err">{error} <button className="btn ghost sm" onClick={() => setError(null)}>×</button></div>}
          {notice && <div className="banner info">{notice} <button className="btn ghost sm" onClick={() => setNotice(null)}>×</button></div>}
          {!wilor?.ready && (
            <div className="banner warn">WiLoR is not installed on this server ({Object.entries(wilor || {}).filter(([k, v]) => k !== 'ready' && !v).map(([k]) => k).join(', ') || 'unknown'} missing): the 3D analysis uses MediaPipe. See the Guide for installing WiLoR and MANO.</div>
          )}
          {stale && <div className="banner warn">Showing a previous result: settings changed{status.k_cached ? ' (display video only)' : ''}. Press “Analyse this video” to update.</div>}

          {video && (
            <div className="videobar">
              <div className="vb-title">
                <b>{video.meta ? `${video.meta.pid} · ${video.meta.session} · ${video.meta.level}` : video.stem}</b>
                <span className="dim">{video.width}×{video.height} · {video.fps.toFixed(0)} fps · {fmtDur(video.duration)}</span>
              </div>
              {evmSeg && settings && (evmSeg.start_s !== settings.start_s || evmSeg.duration_s !== settings.duration_s) && (
                <button className="btn sm" title="Use the segment chosen in the EVM tab"
                  onClick={() => setSettings((s) => ({ ...s, start_s: evmSeg.start_s, duration_s: evmSeg.duration_s }))}>
                  Use EVM segment ({evmSeg.start_s}s + {evmSeg.duration_s}s)
                </button>
              )}
              {history.length > 0 && (
                <select className="variants" value="" onChange={(e) => {
                  const [i, j] = e.target.value.split(':').map(Number)
                  const r = history[i]
                  setSettings({ ...K3.defaults, ...r.params, ...(r.displays[j]?.display || {}) })
                }}>
                  <option value="" disabled>Earlier 3D results ({history.length})</option>
                  {history.map((r, i) => (r.displays.length ? r.displays : [{}]).map((dd, j) => (
                    <option key={`${i}:${j}`} value={`${i}:${j}`}>
                      {r.created.slice(5, 16)} · {r.params.model} · {r.params.start_s}s+{r.params.duration_s}s · {r.params.freq_lo}–{r.params.freq_hi} Hz
                    </option>
                  )))}
                </select>
              )}
            </div>
          )}

          <div id="tour-player">
            <ComparePlayer
              left={leftSrc ? { src: leftSrc, label: showing ? `Segmented hand · ${data.start_s.toFixed(1)}–${(data.start_s + data.duration_s).toFixed(1)} s` : 'Raw video (preview)' } : null}
              right={{ src: null, label: showing ? `3D hand · ${data.model === 'wilor' ? 'WiLoR + MANO' : 'MediaPipe'} · ${data.hand.physical_side} hand` : '3D hand' }}
              rightNode={viewer}
              fps={video?.fps || 30}
              onTime={onTime}
              seekRequest={seekReq}
              initialTime={showing ? 0 : settings?.start_s}
              leftExtra={!showing && sel && (
                <button className="btn sm" onClick={() => setSettings((s) => ({ ...s, start_s: Math.round(rawTime.current * 2) / 2 }))}>Set start here</button>
              )}
              rightTabs={showing && (
                <div className="seg small" title="Hand model shown">
                  <button className={`seg-btn ${ctl.hand === 'skeleton' ? 'on' : ''}`} onClick={() => ctlApi.set('hand', 'skeleton')}>Skeleton</button>
                  {K3.hand_model?.vibeskin && <button className={`seg-btn ${ctl.hand === 'vibeskin' ? 'on' : ''}`} disabled={viewerInfo.vibeskin === false}
                    onClick={() => ctlApi.set('hand', 'vibeskin')} title="VibeSkin VR hand (local only)">VibeSkin</button>}
                  {meta.urls.mesh && <button className={`seg-btn ${ctl.hand === 'mano' ? 'on' : ''}`} onClick={() => ctlApi.set('hand', 'mano')}
                    title="MANO surface fitted by WiLoR (not magnified)">MANO</button>}
                </div>
              )}
              rightPlaceholder={busyHere ? (
                <div className="ph-job">
                  <div className="spinner" />
                  <div>{busyHere.state === 'queued' ? `Waiting in queue (#${busyHere.queue_position})` : busyHere.message}</div>
                  <div className="bar"><div style={{ width: `${(busyHere.progress || 0) * 100}%` }} /></div>
                  <div className="dim mono">{Math.round((busyHere.progress || 0) * 100)}%{busyHere.eta_s != null ? ` · ETA ${Math.round(busyHere.eta_s)}s` : ''}</div>
                </div>
              ) : loading ? <div className="ph-idle"><div className="spinner" /><div>Loading the 3D result…</div></div> : (
                <div className="ph-idle">
                  <div className="big">✋</div>
                  <div>{sel ? 'No 3D result with these settings yet.' : 'Choose a video on the left.'}</div>
                  {sel && <div className="dim">Pick a segment, then press <b>Analyse this video</b>.</div>}
                </div>
              )}
            />
          </div>

          {showing && <ViewerBar ctl={ctl} ctlApi={ctlApi} setCtl={setCtl} data={data} />}

          <div id="tour-timeline">
            <Timeline scan={scanState?.scan} scanState={scanState} duration={video?.duration} segStart={segStart}
              segDur={showing ? data.duration_s : settings?.duration_s} recommended={recommended}
              playhead={showing ? data.start_s + time : time} markers={markers?.rows}
              onPick={(t) => {
                setSettings((s) => ({ ...s, start_s: Math.round(t * 2) / 2 }))
                if (!showing) setSeekReq({ t })
              }} />
          </div>

          <div id="tour-results">
            {showing ? (
              <>
                <div className="result-actions">
                  <span className="dim">3D result {meta.key} · {data.model} · {data.processing_s}s analysis · {data.n_frames} frames</span>
                  <a className="btn sm primary" href={url(meta.urls.bundle)} download>Download all (ZIP)</a>
                  <a className="btn sm" href={url(`${meta.urls.bundle}&videos=false`)} download>Data only</a>
                </div>
                <KinPlots data={data} time={time} onSeek={(t) => setSeekReq({ t })} urls={meta.urls} />
              </>
            ) : (
              <div className="empty-analysis dim">
                The 3D hand, its world / wrist / finger coordinate frames, fingertip motion relative to the wrist, orientation,
                velocity, acceleration, spectra with noise floors, spectrograms and coherence appear here after the analysis.
              </div>
            )}
          </div>
        </section>

        <div className="right" id="tour-settings">
          <SettingsPanel schema={K3.schema} settings={settings} setSettings={setSettings}
            onReset={() => setSettings({ ...K3.defaults, ...(evmSeg || (recommended ? { start_s: recommended.start_s } : {})) })}
            status={status ? { stage_a_cached: status.k_cached, stage_b_cached: status.v_cached } : null}
            onProcess={onProcess} job={busyHere} videoInfo={video} gpu={cfg.compute?.gpu}
            onCancel={() => busyHere && api.cancel(busyHere.id).then(() => api.jobs().then(setJobs))}
            recommended={recommended} presets={PRESETS} onPreset={(p) => setSettings((s) => ({ ...s, ...p.values }))}
            onAutoSegment={() => recommended && setSettings((s) => ({ ...s, start_s: recommended.start_s }))}
            onRunAll={onRunAll} nVideos={videos.length} estimate={estimate3d}
            labels={{ partial: '↻ Display video only (3D kinematics cached)', fastStage: 'display' }} />
          <QueuePanel jobs={jobs} onCancel={(j) => api.cancel(j.id).then(() => api.jobs().then(setJobs))} onOpen={(j) => setSel(j.video)} />
        </div>
      </div>
      <Tour steps={TOUR} open={tour} onClose={() => setTour(false)} />
    </div>
  )
}

function ViewerBar({ ctl, ctlApi, setCtl, data }) {
  const amax = 400
  const toSlider = (a) => Math.log(a) / Math.log(amax)
  const fromSlider = (v) => Math.max(1, Math.round(Math.exp(v * Math.log(amax))))
  const showTremor = () => setCtl((c) => ({ ...c, mode: 'frozen', alpha: autoAlpha(data, 'hybrid'), source: 'hybrid',
    show: { ...c.show, trails: true, ellipsoids: true, head: false } }))
  const raw = () => setCtl((c) => ({ ...c, mode: 'follow', alpha: 1, show: { ...c.show, trails: false, ellipsoids: false } }))
  const Chip = ({ k, label, title }) => (
    <button className={`chip ${ctl.show[k] ? 'on' : ''}`} onClick={() => ctlApi.toggle(k)} title={title}>{label}</button>
  )
  return (
    <div className="viewerbar" id="tour-viewer">
      <div className="vb-group">
        <button className="btn sm primary" onClick={showTremor} title={`Freeze the wrist, magnify finger tremor x${autoAlpha(data, 'hybrid')} (so the tremor ellipsoids are about 8 mm), show trails and ellipsoids`}>Show tremor</button>
        <button className="btn sm" onClick={raw} title="Measured motion, no magnification">Measured</button>
      </div>
      <div className="vb-group" title="Wrist (hand as a whole): follow its measured pose, its slow (low-passed) pose, or freeze it at the mean pose">
        <span className="dim">Wrist</span>
        <div className="seg small">
          {[['follow', 'Follow'], ['slow', 'Slow'], ['frozen', 'Frozen']].map(([k, l]) => (
            <button key={k} className={`seg-btn ${ctl.mode === k ? 'on' : ''}`} onClick={() => ctlApi.set('mode', k)}>{l}</button>
          ))}
        </div>
      </div>
      <div className="vb-group" title="Source of the finger vibration that is magnified">
        <span className="dim">Vibration</span>
        <div className="seg small">
          <button className={`seg-btn ${ctl.source === 'hybrid' ? 'on' : ''}`} onClick={() => ctlApi.set('source', 'hybrid')}>Hybrid</button>
          <button className={`seg-btn ${ctl.source === 'model' ? 'on' : ''}`} onClick={() => ctlApi.set('source', 'model')}>Model</button>
        </div>
      </div>
      <div className="vb-group mag" title={`Gain applied to the ${data.band[0]}-${data.band[1]} Hz finger motion relative to the wrist`}>
        <span className="dim">Magnify</span>
        <input type="range" min={0} max={1} step={0.001} value={toSlider(ctl.alpha)} onChange={(e) => ctlApi.set('alpha', fromSlider(parseFloat(e.target.value)))} />
        <span className="mono">×{ctl.alpha}</span>
      </div>
      <div className="vb-group chips">
        <Chip k="world" label="World" title="World frame axes and floor grid" />
        <Chip k="wrist" label="Wrist frame" title="Wrist (palm) frame: x red, y green (towards the fingers), z blue (back of the hand)" />
        <Chip k="fingers" label="Finger frames" title="Rigid-finger frames: x along the finger" />
        <Chip k="head" label="Head camera" title="Recording camera (head) frame, world mode only" />
        <Chip k="trails" label="Trails" title="Fingertip paths over the last second" />
        <Chip k="ellipsoids" label="Ellipsoids" title="2-sigma tremor ellipsoids of the fingertips (x magnification)" />
        <button className={`chip ${ctl.followCam ? 'on' : ''}`} onClick={() => ctlApi.set('followCam', !ctl.followCam)}
          title="Keep the hand centred in the view">Follow hand</button>
      </div>
    </div>
  )
}
