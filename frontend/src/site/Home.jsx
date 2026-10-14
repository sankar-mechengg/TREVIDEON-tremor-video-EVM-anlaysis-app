import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { APP, CITATION, ORG, PEOPLE, SPECS } from './content.js'
import { LineChart } from '../components/Charts.jsx'
import { ANCHOR_COLORS, api } from '../api.js'

gsap.registerPlugin(ScrollTrigger)
const Hero3D = lazy(() => import('./Hero3D.jsx'))   // three.js loads after the page is visible
const Example3D = lazy(() => import('./Example3D.jsx'))

const STEPS = [
  ['01', 'Record', `Film the hand at ${SPECS.recFps} fps or more (egocentric or fixed camera, bright flicker-free light).`],
  ['02', 'Upload', 'Add one or many MP4 videos (up to 1 GB each) and, optionally, a marker CSV per video.'],
  ['03', 'Segment', 'A quick scan finds where the hand is visible and suggests the steadiest segment.'],
  ['04', 'Isolate', 'Optional: cut out the dominant hand and pin the wrist so voluntary movement disappears.'],
  ['05', 'Magnify', 'Phase-based Eulerian Video Magnification makes sub-pixel tremor visible (x15).'],
  ['06', 'Measure', 'Optical flow on the un-magnified video gives frequency, amplitude, spectra, envelopes.'],
  ['07', '3D', 'Optional: rebuild the hand in 3D (WiLoR / MediaPipe) and measure each finger relative to the wrist.'],
  ['08', 'Export', 'Download videos, metrics, spectra and marker-segment statistics as one ZIP.'],
]

const FAQ = [
  ['Is my video uploaded to a server?', `Only when you use the hosted version, and only to analyse it: files are kept in temporary storage for at most ${SPECS.ttlMin} minutes after your last activity, or deleted immediately with “Delete my data”. Nothing is used for anything else. When you run ${APP.name} on your own computer, nothing leaves it.`],
  ['Does the magnification create the tremor it shows?', 'No. All numbers come from optical flow on the original (un-magnified) video. Magnification is only a visual aid; turning α up cannot create tremor in the results.'],
  ['Why 100 fps?', 'Physiological tremor lies at 8–12 Hz. A camera can only represent rhythms below half its frame rate, so 30 fps stops at 15 Hz and smears fast motion; 100 fps or more resolves the whole tremor band with margin and less motion blur.'],
  ['What does “hand isolation” do?', 'It segments the dominant hand (SAM 2.1 on a GPU, MediaPipe on CPU), removes the background, scales every hand to the same size and pins the wrist centre (+). Voluntary arm movement is removed; tremor relative to the wrist remains, comparable across people.'],
  ['Can I use it for diagnosis?', 'No. TREVIDEON is a research tool and not a medical device. Results must not be used for diagnosis or treatment decisions.'],
  ['My video repeats frames. Is that a problem?', 'Screen / VR recordings and variable-frame-rate phones often repeat a frame regularly, which creates false spectral peaks (e.g. 2.5, 5, 10 Hz). TREVIDEON detects repeated frames and replaces them with motion-interpolated frames, and reports how many it repaired.'],
]

function Reveal({ children, as: Tag = 'div', className = '', ...rest }) {
  return <Tag className={`reveal ${className}`} {...rest}>{children}</Tag>
}

function Example() {
  const [a, setA] = useState(null)
  const vids = useRef([])
  const base = '/examples/p06-high'
  useEffect(() => { fetch(`${base}/analysis.json`).then((r) => r.json()).then(setA).catch(() => setA(null)) }, [])
  useEffect(() => {
    const master = vids.current[0]
    if (!master) return
    let raf
    const sync = () => {
      vids.current.slice(1).forEach((v) => {
        if (v && Math.abs(v.currentTime - master.currentTime) > 0.08) v.currentTime = master.currentTime
      })
      raf = requestAnimationFrame(sync)
    }
    raf = requestAnimationFrame(sync)
    return () => cancelAnimationFrame(raf)
  }, [a])
  const spectrum = useMemo(() => {
    if (!a) return []
    const f = a.spectrum.freqs
    return ['thumb', 'index', 'middle', 'ring', 'pinky'].map((k) => ({
      name: k, color: ANCHOR_COLORS[k], x: f, y: a.spectrum.psd[k].map((p, i) => p * (2 * Math.PI * f[i]) ** 2), width: 1.5,
    }))
  }, [a])
  const play = () => vids.current.forEach((v) => { if (v) { v.currentTime = vids.current[0].currentTime; v.play().catch(() => {}) } })
  return (
    <div className="example">
      <div className="ex-videos" onClick={play}>
        {[['original.mp4', 'Raw video (segment)'], ['magnified.mp4', 'Eulerian magnification ×15'], ['overlay.mp4', 'Tremor-envelope overlay']].map(([f, l], i) => (
          <figure key={f}>
            <video ref={(el) => { vids.current[i] = el }} src={`${base}/${f}`} muted loop playsInline autoPlay preload="auto" />
            <figcaption>{l}</figcaption>
          </figure>
        ))}
      </div>
      {a && (
        <div className="ex-data">
          <div className="ex-cards">
            <div><span>Dominant frequency</span><b>{a.metrics.hand.peak_hz} Hz</b></div>
            <div><span>Whole-hand RMS</span><b>{a.metrics.hand.rms_px.toFixed(2)} px</b></div>
            <div><span>Tremor-band power</span><b>{a.metrics.hand.band_power_pct.toFixed(1)} %</b></div>
            <div><span>Hand tracked</span><b>{a.hand_detected_pct.toFixed(0)} %</b></div>
            <div><span>Repeated frames repaired</span><b>{a.frame_repair?.repaired_frames ?? 0}</b></div>
          </div>
          <div className="ex-chart">
            <LineChart series={spectrum} logY xDomain={[1, a.fps / 2]} band={a.band} height={210}
              xLabel="frequency (Hz)" yLabel="velocity PSD" legend />
          </div>
          <p className="note">
            Example: the developer's own hand ({a.duration_s.toFixed(0)} s, egocentric VR-headset recording at {a.fps.toFixed(0)} fps,
            so the analysable band ends below {(a.fps / 2).toFixed(0)} Hz). Headset screen recordings repeat and drop frames irregularly;
            {APP.name} repairs the repeats ({a.frame_repair?.repaired_frames ?? 0} here), but the regularly spaced peaks that remain (about
            2.5 Hz apart) come from that frame timing, not from the hand. A camera recording at a constant {SPECS.recFps} fps or more avoids
            this and resolves the full tremor band. Click the videos to restart them together.
          </p>
        </div>
      )}
    </div>
  )
}

export default function Home() {
  const nav = useNavigate()
  const hero = useRef(null)
  const root = useRef(null)
  const [copied, setCopied] = useState(false)
  const [launching, setLaunching] = useState(false)

  useEffect(() => {
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches || new URLSearchParams(window.location.search).has('still')
    const ctx = gsap.context(() => {
      if (!reduced) {
        gsap.from('.hero-copy > *', { y: 26, opacity: 0, duration: 0.9, stagger: 0.12, ease: 'power3.out', delay: 0.3 })
        gsap.utils.toArray('.reveal').forEach((el) => {
          gsap.from(el, { y: 36, opacity: 0, duration: 0.8, ease: 'power2.out',
            scrollTrigger: { trigger: el, start: 'top 85%' } })
        })
        gsap.from('.step', { y: 30, opacity: 0, stagger: 0.08, duration: 0.6,
          scrollTrigger: { trigger: '.steps', start: 'top 80%' } })
      }
    }, root)
    return () => ctx.revert()
  }, [])

  const launch = () => {
    if (launching) return
    setLaunching(true)
    if (hero.current) hero.current.launch(() => nav('/analyze'))
    else nav('/analyze')
  }

  return (
    <div className="home" ref={root}>
      <section className="hero">
        <Suspense fallback={<div className="hero3d" />}><Hero3D ref={hero} /></Suspense>
        <div className="portal-flash" />
        <div className="hero-copy">
          <p className="eyebrow">Open-source · Eulerian Video Magnification · Hand tremor</p>
          <h1>{APP.name}</h1>
          <p className="hero-exp">{APP.expansion}</p>
          <p className="hero-tag">{APP.tagline} Upload a hand video, magnify its micro-motion and measure the tremor:
            frequency, amplitude, spectra and how it changes over time.</p>
          <div className="hero-cta">
            <button className="btn primary xl" onClick={launch}>Launch analysis →</button>
            <a className="btn ghost-light xl" href="#example">See an example</a>
            <Link className="btn ghost-light xl" to="/guide">How to record</Link>
          </div>
          <p className="hero-note">Move your mouse over the hand: inside the lens the tremor is magnified ×14.</p>
        </div>
        <div className="scroll-hint" aria-hidden="true">scroll</div>
      </section>

      <section className="sec">
        <Reveal as="h2">What {APP.name} does</Reveal>
        <div className="cards3">
          <Reveal className="fcard"><h3>Makes tremor visible</h3><p>Hand tremor moves the skin by a fraction of a pixel. Phase-based Eulerian Video Magnification (Riesz pyramid) amplifies only the 4–12 Hz rhythm, so you can watch it.</p></Reveal>
          <Reveal className="fcard"><h3>Measures it honestly</h3><p>Dense optical flow on the original video gives per-finger frequency, RMS amplitude, band power, envelopes and spectrograms. The numbers never depend on the magnification.</p></Reveal>
          <Reveal className="fcard"><h3>Links it to events</h3><p>Add a marker CSV (conditions, faults, stress ratings) and get tremor statistics for every marked segment, ready for stress–tremor correlation.</p></Reveal>
        </div>
      </section>

      <section className="sec alt">
        <Reveal as="h2">How it works</Reveal>
        <div className="steps">
          {STEPS.map(([n, t, d]) => (
            <div className="step" key={n}><span className="step-n">{n}</span><h4>{t}</h4><p>{d}</p></div>
          ))}
        </div>
      </section>

      <section className="sec" id="example">
        <Reveal as="h2">Example: raw video and results</Reveal>
        <Reveal as="p" className="lead">The same 12-second segment, shown three ways and measured. The overlay colours how strongly each part of the hand trembles; fingertip trails show the shape of the shake.</Reveal>
        <Reveal><Example /></Reveal>
      </section>

      <section className="sec alt" id="kin3d">
        <Reveal as="h2">3D kinematics: each finger&apos;s own tremor</Reveal>
        <Reveal as="p" className="lead">The hand is rebuilt in 3D in every frame and described by coordinate frames: the world, the wrist, and one rigid frame per finger. A finger&apos;s pose relative to the wrist removes the motion of the hand as a whole, leaving that finger&apos;s own vibration. It is measured with sub-pixel optical flow in the wrist frame and checked against a measured noise floor. Below, the wrist is frozen and only the finger tremor is magnified.</Reveal>
        <Reveal><Suspense fallback={<div className="ex3d-loading">Loading the 3D example…</div>}><Example3D /></Suspense></Reveal>
        <Reveal className="center-cta"><Link className="btn primary" to="/analyze/3d">Open the 3D tab →</Link></Reveal>
      </section>

      <section className="sec" id="record">
        <Reveal as="h2">Record it right</Reveal>
        <div className="rec-grid">
          <Reveal className="rec-card big">
            <div className="spec-big">≥ {SPECS.recFps} fps</div>
            <p>Minimum recommended frame rate. Tremor peaks at 8–12 Hz; a camera only sees rhythms below half its frame rate. Videos at 30–99 fps are accepted with a warning, below {SPECS.minFps} fps they are refused.</p>
          </Reveal>
          <Reveal className="rec-card"><h4>Cameras</h4><p>Action cameras (DJI Osmo Action, GoPro, Insta360) at 4K / 100–120 fps · USB webcams that really deliver 100–120 fps (usually at 720p / 1080p) · phone slow-motion modes at 120–240 fps (export at the capture rate).</p></Reveal>
          <Reveal className="rec-card"><h4>Egocentric or fixed</h4><p>Head- or chest-mounted (egocentric) views work; a tripod is best. Keep the whole hand in view, about a third of the frame height, fingers spread, for at least 20 s.</p></Reveal>
          <Reveal className="rec-card"><h4>Light</h4><p>Bright, even, flicker-free light. In 50 Hz countries (e.g. India) ordinary lamps flicker at 100 Hz, which interferes with 100 fps recordings: use daylight or flicker-free LED panels and a shutter of 1/250–1/500 s.</p></Reveal>
          <Reveal className="rec-card"><h4>File</h4><p>MP4 (H.264 / H.265), constant frame rate, up to 1 GB per video. Avoid screen recordings where possible; if you must, {APP.name} repairs repeated frames.</p></Reveal>
        </div>
        <Reveal className="center-cta"><Link className="btn primary" to="/guide">Full recording guide →</Link></Reveal>
      </section>

      <section className="sec alt" id="markers">
        <Reveal as="h2">Marker CSV (optional)</Reveal>
        <div className="two">
          <Reveal>
            <p className="lead">Tell {APP.name} what happened when: one row per event, times from the start of the video. Rows with an end time are segments; tremor metrics are computed for each.</p>
            <ul className="ticks">
              <li><b>time_s</b> (required): seconds (12.5) or m:ss (0:12.5)</li>
              <li><b>end_s</b>: end of a segment (empty for a single event)</li>
              <li><b>marker</b> (required): CONDITION, BASELINE, FAULT, STRESS_RATING, …</li>
              <li><b>value</b>, <b>notes</b>: free text or numbers</li>
            </ul>
            <div className="row-btns">
              <a className="btn primary" href={api.templateUrl()} download>Download template</a>
              <Link className="btn" to="/markers">Specification →</Link>
            </div>
          </Reveal>
          <Reveal>
            <pre className="code">{`time_s,end_s,marker,value,notes
0.0,,SESSION_START,,video starts
5.0,65.0,BASELINE,,hand held still
70.0,130.0,CONDITION,Low,low-difficulty block
75.4,,FAULT,,ring touched the wire
131.0,,STRESS_RATING,3,self-report 1-10`}</pre>
          </Reveal>
        </div>
      </section>

      <section className="sec">
        <Reveal as="h2">What you get</Reveal>
        <div className="cards4">
          {[['Videos', 'Segment, magnified (×α) and tremor-overlay MP4s, synchronised in the player'],
            ['Tremor metrics', 'Peak frequency, RMS / P95 amplitude, band power and % of hand size per finger and whole hand'],
            ['Time & frequency', 'Spectra, sliding-RMS envelopes, spectrograms and fingertip micro-trajectories'],
            ['Marker statistics', 'Per-segment / per-event metrics as CSV, plus everything in one ZIP']].map(([t, d]) => (
            <Reveal className="fcard" key={t}><h3>{t}</h3><p>{d}</p></Reveal>
          ))}
        </div>
      </section>

      <section className="sec alt" id="privacy">
        <div className="two">
          <Reveal>
            <h2>Nothing is kept</h2>
            <p className="lead">No accounts, no tracking, no analytics. On the hosted version your videos live in temporary storage for the analysis only and are deleted {SPECS.ttlMin} minutes after your last activity, when you click “Delete my data”, or when the server restarts. Run {APP.name} on your own computer and nothing leaves it.</p>
            <Link className="btn" to="/privacy">Privacy policy →</Link>
          </Reveal>
          <Reveal>
            <h2>Open source</h2>
            <p className="lead">{APP.name} is released under the {APP.license}. Use it, study it, improve it. Contributions are welcome.</p>
            <div className="row-btns">
              <a className="btn" href={APP.repo} target="_blank" rel="noreferrer">Source code ↗</a>
              <Link className="btn" to="/about#license">License</Link>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="sec" id="cite">
        <Reveal as="h2">How to cite</Reveal>
        <Reveal as="p" className="lead">If {APP.name} helps your research, please cite the software (and the methods it builds on, listed on the Cite page).</Reveal>
        <Reveal>
          <p className="cite-text">{CITATION.text}</p>
          <div className="code-wrap">
            <pre className="code">{CITATION.bibtex}</pre>
            <button className="btn sm copy" onClick={() => { navigator.clipboard?.writeText(CITATION.bibtex); setCopied(true); setTimeout(() => setCopied(false), 1500) }}>
              {copied ? 'Copied' : 'Copy BibTeX'}
            </button>
          </div>
          <p className="dim">{CITATION.doiNote}</p>
        </Reveal>
      </section>

      <section className="sec alt">
        <Reveal as="h2">Who made it</Reveal>
        <div className="team">
          <Reveal className="person"><b>{PEOPLE.developer.name}</b><span>Developer</span>
            <a href={PEOPLE.developer.linkedin} target="_blank" rel="noreferrer">linkedin.com/in/sankar4</a></Reveal>
          <Reveal className="person"><b>{PEOPLE.pi.name}</b><span>Principal Investigator</span>
            <a href={ORG.lab.url} target="_blank" rel="noreferrer">RISE Lab</a></Reveal>
          <Reveal className="person org"><img src={ORG.lab.logo} alt="RISE Lab" className="logo-rise" />
            <span>{ORG.lab.full}<br />{ORG.centre.full}, {ORG.institute.full}</span></Reveal>
        </div>
      </section>

      <section className="sec">
        <Reveal as="h2">Questions</Reveal>
        <div className="faq">
          {FAQ.map(([q, a]) => (
            <Reveal as="details" key={q}><summary>{q}</summary><p>{a}</p></Reveal>
          ))}
        </div>
        <Reveal className="center-cta"><button className="btn primary xl" onClick={() => { window.scrollTo({ top: 0, behavior: 'smooth' }); setTimeout(launch, 700) }}>Start analysing →</button></Reveal>
      </section>
    </div>
  )
}
