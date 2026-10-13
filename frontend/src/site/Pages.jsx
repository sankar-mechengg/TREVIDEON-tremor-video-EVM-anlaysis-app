import { useState } from 'react'
import { Link } from 'react-router-dom'
import { APP, CITATION, METHOD_REFS, ORG, PEOPLE, SPECS } from './content.js'
import { api } from '../api.js'

function Page({ title, kicker, children }) {
  return (
    <article className="page">
      <header className="page-head">
        {kicker && <p className="eyebrow">{kicker}</p>}
        <h1>{title}</h1>
      </header>
      <div className="page-body">{children}</div>
    </article>
  )
}

export function Guide() {
  return (
    <Page title="Recording guide" kicker="Getting good tremor videos">
      <p className="lead">{APP.name} measures motion of a fraction of a pixel. The quality of the recording decides what can be measured. Follow these points and the hand tremor will be resolved reliably.</p>

      <h2 id="fps">1. Frame rate: at least {SPECS.recFps} fps</h2>
      <p>Physiological tremor lies around 8–12 Hz; pathological tremors are mostly 3–12 Hz. A video can only represent rhythms below half its frame rate (the Nyquist limit), and fast motion is smeared within each frame. Therefore:</p>
      <table className="tbl">
        <thead><tr><th>Frame rate</th><th>Highest usable frequency</th><th>{APP.name}</th></tr></thead>
        <tbody>
          <tr><td>≥ 100 fps (recommended)</td><td>≥ 45 Hz</td><td><span className="pill ok">ideal</span> whole tremor band with margin, little motion blur</td></tr>
          <tr><td>60 fps</td><td>27 Hz</td><td><span className="pill warn">accepted</span> band limited, more blur</td></tr>
          <tr><td>30 fps</td><td>13.5 Hz</td><td><span className="pill warn">accepted with warning</span> band ends just above 12 Hz</td></tr>
          <tr><td>&lt; {SPECS.minFps} fps</td><td>&lt; 11 Hz</td><td><span className="pill bad">refused</span></td></tr>
        </tbody>
      </table>
      <p>Record at a <b>constant</b> frame rate and export at the capture rate (not “slowed down” to 30 fps). Phones in slow-motion mode often save 120/240 fps footage re-timed to 30 fps playback; export it at the original frame rate (check the file's frame rate in a media player before uploading).</p>

      <h2 id="cameras">2. Cameras that work</h2>
      <ul>
        <li><b>Action cameras</b>: DJI Osmo Action, GoPro HERO, Insta360 and similar record 4K at 100–120 fps and 1080p at 200–240 fps. Good for egocentric mounting (head strap, chest mount).</li>
        <li><b>High-frame-rate webcams</b>: USB 3 / UVC cameras that genuinely deliver 100–120 fps (most do so at 720p or 1080p; many “4K webcams” only reach 30 fps at 4K). Check the frame rate actually recorded, not the box.</li>
        <li><b>Phones</b>: slow-motion modes (120/240 fps). Lock exposure and focus; disable stabilisation if it warps the frame.</li>
        <li><b>Resolution</b>: 1080p is plenty; 4K helps when the hand is small in the frame. {APP.name} works at 720p by default (configurable).</li>
      </ul>

      <h2 id="setup">3. Set-up and posture</h2>
      <ul>
        <li><b>Egocentric</b> (camera on the head or chest, looking at your own hand) or <b>fixed</b> (tripod). A tripod gives the cleanest data; with head-mounted cameras the background stabilisation removes most head motion.</li>
        <li>Keep the <b>whole hand</b> in view, roughly a third of the frame height, back of the hand or palm facing the camera, fingers slightly spread.</li>
        <li>Record at least <b>20–30 s</b> of each condition (longer is better: spectra average more cycles). Add a few seconds before and after.</li>
        <li>Use a plain, textured background (a table top or floor is fine); avoid moving people or screens behind the hand.</li>
        <li>Wrist-worn devices are fine, but their screens and cables create spurious motion: rely on the fingertips.</li>
      </ul>

      <h2 id="light">4. Lighting and shutter</h2>
      <ul>
        <li>Bright, even light. At 100 fps each frame gets at most 1/100 s, so dim scenes become noisy.</li>
        <li><b>Flicker</b>: lamps on 50 Hz mains (India, Europe) flicker at 100 Hz, which beats with 100 fps recordings and produces brightness waves. Use daylight, DC / “flicker-free” LED panels, or a shutter that is a multiple of 1/100 s. On 60 Hz mains the same applies to 120 Hz.</li>
        <li>Shutter 1/250–1/500 s keeps fingers sharp; raise the light rather than the ISO.</li>
      </ul>

      <h2 id="protocol">5. A simple protocol</h2>
      <ol>
        <li>Start recording; hold the hand still in view for 5 s (and note the time: this is a natural <code>BASELINE</code> marker).</li>
        <li>Perform the task or condition; note start / end times in the <Link to="/markers">marker CSV</Link>.</li>
        <li>Keep one recording per condition, or mark the conditions in one long recording.</li>
        <li>Check the file: frame rate, length, the hand visible throughout.</li>
      </ol>

      <h2 id="consent">6. Consent and privacy</h2>
      <p>Only record and upload videos of people who have given their informed consent (and, for research, under the approval of your ethics committee). Prefer framing that shows only the hand.</p>

      <div className="center-cta"><Link className="btn primary" to="/analyze">Analyse a video →</Link></div>
    </Page>
  )
}

export function Markers() {
  return (
    <Page title="Marker CSV specification" kicker="Link tremor to events">
      <p className="lead">A marker file is optional. It lists what happened during the video so that {APP.name} can compute tremor statistics for each condition or around each event, e.g. for stress–tremor correlation.</p>
      <div className="row-btns"><a className="btn primary" href={api.templateUrl()} download>Download the template</a></div>
      <h2>Format</h2>
      <p>UTF-8 CSV with a header row. One row per event. Times are measured from the <b>start of the video file</b>.</p>
      <table className="tbl">
        <thead><tr><th>Column</th><th>Required</th><th>Content</th></tr></thead>
        <tbody>
          <tr><td><code>time_s</code></td><td>yes</td><td>Event time: seconds (<code>12.5</code>) or <code>m:ss(.ms)</code> / <code>h:mm:ss(.ms)</code></td></tr>
          <tr><td><code>end_s</code></td><td>no</td><td>End time (same formats). A row with an end is a <b>segment</b>; without, an <b>event</b>.</td></tr>
          <tr><td><code>marker</code></td><td>yes</td><td>Label. Suggested: <code>SESSION_START</code>, <code>SESSION_END</code>, <code>BASELINE</code>, <code>CONDITION</code>, <code>TASK</code>, <code>FAULT</code>, <code>STRESS_RATING</code>, <code>NOTE</code>. Any other label is accepted.</td></tr>
          <tr><td><code>value</code></td><td>no</td><td>Free value: condition name (Low / High), a rating (7), …</td></tr>
          <tr><td><code>notes</code></td><td>no</td><td>Free text.</td></tr>
        </tbody>
      </table>
      <h2>Example</h2>
      <pre className="code">{`time_s,end_s,marker,value,notes
0.0,,SESSION_START,,video starts
5.0,65.0,BASELINE,,hand held still - no task
70.0,130.0,CONDITION,Low,low-difficulty block
75.4,,FAULT,,ring touched the wire
131.0,,STRESS_RATING,3,self-report 1-10
140.0,200.0,CONDITION,High,high-difficulty block
152.8,,FAULT,,ring touched the wire
201.0,,STRESS_RATING,8,self-report 1-10
205.0,,SESSION_END,,`}</pre>
      <h2>What is computed</h2>
      <ul>
        <li>For every <b>segment</b>: the part that overlaps the analysed video segment, and inside it the fingertip RMS tremor (px and % of hand length), the mean whole-hand envelope and the dominant frequency.</li>
        <li>For every <b>event</b>: the same metrics in a 2-second window centred on the event.</li>
        <li>Markers are drawn on the timeline and exported as <code>marker_metrics.csv</code> in the result ZIP.</li>
      </ul>
      <h2>Validation</h2>
      <p>On upload, {APP.name} checks the header, every time and the order of start / end, and reports problems with their line numbers. Events after the end of the video produce a warning.</p>
    </Page>
  )
}

export function Cite() {
  const [copied, setCopied] = useState('')
  const copy = (k, t) => { navigator.clipboard?.writeText(t); setCopied(k); setTimeout(() => setCopied(''), 1500) }
  return (
    <Page title="How to cite" kicker="Credit the work you build on">
      <p className="lead">If you use {APP.name} or its results in a publication, presentation or product, please cite the software. Please also cite the methods it implements.</p>
      <h2>Software</h2>
      <p className="cite-text">{CITATION.text}</p>
      <div className="code-wrap">
        <pre className="code">{CITATION.bibtex}</pre>
        <button className="btn sm copy" onClick={() => copy('bib', CITATION.bibtex)}>{copied === 'bib' ? 'Copied' : 'Copy BibTeX'}</button>
      </div>
      <p className="dim">{CITATION.doiNote} A machine-readable <code>CITATION.cff</code> is included in the repository.</p>
      <h2>Methods</h2>
      <ul className="refs">
        {METHOD_REFS.map((r) => <li key={r.key}><b>{r.key}</b> ({r.use}): {r.text}</li>)}
      </ul>
      <h2>Suggested methods text</h2>
      <p className="quote">Hand tremor was quantified with {APP.name} v{APP.version} (Balasubramanian & Mukherjee, {APP.year}). Repeated video frames were repaired by motion interpolation. Camera motion above 1 Hz was removed by background-feature stabilisation. Sub-pixel motion of the hand was measured with dense optical flow (DIS; Kroeger et al., 2016) on the un-magnified video, band-passed to 4–12 Hz, and summarised as per-finger dominant frequency (Welch velocity spectrum), RMS amplitude and band power. For visualisation, motion was magnified 15× with phase-based Eulerian Video Magnification (Riesz pyramid; Wadhwa et al., 2014). Hand landmarks were obtained with MediaPipe Hands (Zhang et al., 2020).</p>
    </Page>
  )
}

export function Privacy() {
  return (
    <Page title="Privacy policy" kicker={`Effective ${APP.year}-10-13 · version 1`}>
      <p className="lead">{APP.name} is built to analyse videos without keeping them. This policy explains what happens to your data.</p>
      <h2>1. Who is responsible</h2>
      <p>{APP.name} is developed at {ORG.lab.full} ({ORG.lab.name}), {ORG.centre.full}, {ORG.institute.full}. Contact: via <a href={ORG.lab.url} target="_blank" rel="noreferrer">{ORG.lab.url}</a>.</p>
      <h2>2. Running {APP.name} on your own computer</h2>
      <p>The local version processes everything on your computer. No video, result or usage information is sent anywhere.</p>
      <h2>3. Using the hosted website</h2>
      <ul>
        <li><b>What we process</b>: the videos and marker files you upload, the settings you choose, and the results computed from them. The server also sees technical connection data (IP address, browser type) as part of normal web traffic.</li>
        <li><b>Why</b>: only to perform the analysis you request and to show you the results.</li>
        <li><b>Where and how long</b>: in temporary storage on the analysis server, under a random session identifier known only to your browser tab. Everything is deleted {SPECS.ttlMin} minutes after your last activity, immediately when you click “Delete my data”, about two minutes after you close the tab (best effort; a page reload keeps your session), and whenever the server restarts. There are no backups of uploaded content.</li>
        <li><b>What we do not do</b>: no accounts, no advertising, no analytics or tracking cookies, no sharing or selling of data, no use of your videos to train models, no human review of uploads.</li>
        <li><b>Logs</b>: server logs may record technical events (time, request path, errors) for operation and security; they do not contain your video content and are kept for a short period.</li>
        <li><b>Processors</b>: the website is served by a static hosting provider and the analysis runs on a cloud compute provider; they process data only to provide these services.</li>
      </ul>
      <h2>4. Videos of people</h2>
      <p>Videos of hands can be personal data, and some may show faces or other identifying details. Upload only videos you are entitled to process: your own, or of people who gave informed consent (and, for research, with ethics approval). Crop or frame recordings to the hand where possible.</p>
      <h2>5. Children</h2>
      <p>The hosted service is not intended for use by children without the involvement of a parent or guardian.</p>
      <h2>6. Your rights</h2>
      <p>Because content is deleted automatically and no accounts exist, there is normally nothing left to access or erase after your session; you can delete your session at any time with “Delete my data”. For questions contact the lab.</p>
      <h2>7. Changes</h2>
      <p>Changes to this policy will be published on this page with a new effective date.</p>
      <p className="dim">This policy is a template prepared for the research tool and should be reviewed by IIT Delhi before public launch.</p>
    </Page>
  )
}

export function Terms() {
  return (
    <Page title="Terms of use" kicker={`Effective ${APP.year}-10-13 · version 1`}>
      <h2>1. The software</h2>
      <p>{APP.name} is open-source software released under the <Link to="/about#license">{APP.license}</Link>. The license governs your use, copying, modification and distribution of the code. These terms additionally govern the use of the hosted website.</p>
      <h2 id="disclaimer">2. Research use only, not a medical device</h2>
      <p>{APP.name} is a research tool. It is <b>not a medical device</b>, has not been validated or approved for clinical use, and must not be used to diagnose, monitor or treat any disease or to make clinical decisions. Results are estimates of image motion and depend on recording conditions.</p>
      <h2>3. Your content and responsibilities</h2>
      <ul>
        <li>You may upload only videos and files you have the right to process, including the informed consent of every person recorded.</li>
        <li>You must not upload unlawful content, malware, or content unrelated to hand-motion analysis, and must not attempt to disrupt, overload or reverse-engineer the hosted service.</li>
        <li>You keep all rights to your content. You grant us only the permission needed to process it for your request; it is deleted as described in the <Link to="/privacy">privacy policy</Link>.</li>
      </ul>
      <h2>4. Availability and limits</h2>
      <p>The hosted service is provided free of charge, on a best-effort basis, with limits (currently up to {SPECS.maxUploadGB} GB per video and {SPECS.hostedSegment} s analysed per video at up to {SPECS.hostedHeight}p). It may be changed, interrupted or discontinued at any time. For unlimited or confidential processing, run {APP.name} on your own computer.</p>
      <h2>5. No warranty, limitation of liability</h2>
      <p>The software and service are provided “as is”, without warranties of any kind, express or implied, including fitness for a particular purpose and accuracy of results. To the extent permitted by law, the authors and their institutions are not liable for any damages arising from the use of the software, the service or the results.</p>
      <h2>6. Attribution</h2>
      <p>Please cite {APP.name} when you use it in research (see <Link to="/cite">How to cite</Link>). The names and logos of IIT Delhi and RISE Lab identify where the software was developed and may not be used to suggest endorsement of your work.</p>
      <h2>7. Governing law</h2>
      <p>These terms are governed by the laws of India; courts at New Delhi have jurisdiction.</p>
      <p className="dim">These terms are a template prepared for the research tool and should be reviewed by IIT Delhi before public launch.</p>
    </Page>
  )
}

export function About() {
  return (
    <Page title={`About ${APP.name}`} kicker={APP.expansion}>
      <p className="lead">{APP.name} grew out of the VibeSkin project at {ORG.lab.name}, which studies how stress changes hand tremor during a precision task. It packages the video side of that work, Eulerian Video Magnification plus optical-flow tremor measurement, as an open tool anyone can use.</p>
      <h2>Team</h2>
      <ul>
        <li><b>{PEOPLE.developer.name}</b>, developer · <a href={PEOPLE.developer.linkedin} target="_blank" rel="noreferrer">linkedin.com/in/sankar4</a></li>
        <li><b>{PEOPLE.pi.name}</b>, principal investigator, {ORG.lab.name}</li>
      </ul>
      <div className="org-logos big">
        <a href={ORG.lab.url} target="_blank" rel="noreferrer"><img src={ORG.lab.logo} alt="RISE Lab logo" className="logo-rise" /></a>
        <a href={ORG.institute.url} target="_blank" rel="noreferrer"><img src={ORG.institute.logo} alt="IIT Delhi logo" className="logo-iitd" /></a>
      </div>
      <p>{ORG.lab.full}, {ORG.centre.full}, {ORG.institute.full}.</p>
      <h2 id="license">License</h2>
      <p>Copyright {APP.year} {PEOPLE.developer.name}, RISE Lab, IIT Delhi. Licensed under the Apache License, Version 2.0 (the “License”); you may not use this software except in compliance with the License. You may obtain a copy of the License at <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noreferrer">apache.org/licenses/LICENSE-2.0</a>. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” basis, without warranties or conditions of any kind, either express or implied.</p>
      <h2>Third-party components</h2>
      <table className="tbl">
        <thead><tr><th>Component</th><th>Use</th><th>License</th></tr></thead>
        <tbody>
          <tr><td>MediaPipe (hand landmarker, multiclass segmenter)</td><td>hand landmarks, CPU segmentation</td><td>Apache-2.0</td></tr>
          <tr><td>SAM 2.1 (Meta)</td><td>GPU hand segmentation</td><td>Apache-2.0</td></tr>
          <tr><td>OpenCV</td><td>optical flow, video I/O</td><td>Apache-2.0</td></tr>
          <tr><td>PyTorch</td><td>GPU magnification</td><td>BSD-3-Clause</td></tr>
          <tr><td>NumPy, SciPy</td><td>signal processing</td><td>BSD-3-Clause</td></tr>
          <tr><td>FastAPI / Starlette</td><td>API server</td><td>MIT / BSD-3-Clause</td></tr>
          <tr><td>FFmpeg (imageio-ffmpeg)</td><td>video encoding</td><td>LGPL / GPL builds</td></tr>
          <tr><td>React, React Router, Vite</td><td>website</td><td>MIT</td></tr>
          <tr><td>three.js</td><td>3-D home animation</td><td>MIT</td></tr>
          <tr><td>GSAP</td><td>animation</td><td>GSAP Standard “No Charge” License</td></tr>
        </tbody>
      </table>
      <p className="dim">The RISE Lab and IIT Delhi logos are trademarks of their owners and are not covered by the Apache license.</p>
    </Page>
  )
}

export function NotFound() {
  return (
    <Page title="Page not found">
      <p>That page does not exist. <Link to="/">Back to the home page</Link>.</p>
    </Page>
  )
}
