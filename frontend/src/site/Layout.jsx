import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { APP, CITATION, ORG, PEOPLE } from './content.js'

/** TREVIDEON mark: a magnifier lens; the wave grows inside it (tremor made visible). */
export function LogoMark({ size = 30 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true" className="logomark">
      <defs>
        <linearGradient id="lm-g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#22d3ee" />
          <stop offset="1" stopColor="#a78bfa" />
        </linearGradient>
      </defs>
      <circle cx="17" cy="17" r="12" fill="none" stroke="url(#lm-g)" strokeWidth="2.6" />
      <path d="M27 27 L36 36" stroke="url(#lm-g)" strokeWidth="3.4" strokeLinecap="round" />
      <path d="M2 17 h5 l2 -2 2 2 2 -7 3 14 3 -12 2 7 2 -2 h5 h5" fill="none" stroke="#e6ebf5" strokeWidth="1.8"
        strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

const NAV = [
  ['/', 'Home'],
  ['/analyze', 'Analyse'],
  ['/guide', 'Recording guide'],
  ['/markers', 'Marker CSV'],
  ['/cite', 'Cite'],
  ['/about', 'About'],
]

export function Header({ compact = false }) {
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  useEffect(() => setOpen(false), [loc.pathname])
  return (
    <header className={`site-header ${compact ? 'compact' : ''}`}>
      <Link to="/" className="site-brand" aria-label={`${APP.name} home`}>
        <LogoMark size={compact ? 26 : 30} />
        <span className="wordmark">{APP.name}</span>
        {!compact && <span className="brand-sub">{APP.expansion}</span>}
      </Link>
      <button className="nav-toggle" aria-label="Menu" onClick={() => setOpen((o) => !o)}>☰</button>
      <nav className={`site-nav ${open ? 'open' : ''}`}>
        {NAV.map(([to, label]) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'on' : '')}>{label}</NavLink>
        ))}
        <a href={APP.repo} target="_blank" rel="noreferrer" className="gh" title="Source code (Apache-2.0)">GitHub ↗</a>
      </nav>
    </header>
  )
}

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="foot-grid">
        <div className="foot-brand">
          <div className="foot-logo"><LogoMark size={28} /><b>{APP.name}</b></div>
          <p className="dim">{APP.expansion}. {APP.short}.</p>
          <p className="made">
            Made with <span className="heart" aria-label="love">♥</span> by{' '}
            <a href={PEOPLE.developer.linkedin} target="_blank" rel="noreferrer">{PEOPLE.developer.creditName}</a>
          </p>
          <p className="dim">PI: {PEOPLE.pi.name}</p>
        </div>
        <div className="foot-org">
          <p>Created at</p>
          <div className="org-logos">
            <a href={ORG.lab.url} target="_blank" rel="noreferrer" title={`${ORG.lab.name} - ${ORG.lab.full}`}>
              <img src={ORG.lab.logo} alt="RISE Lab logo" className="logo-rise" />
            </a>
            <a href={ORG.institute.url} target="_blank" rel="noreferrer" title={ORG.institute.full}>
              <img src={ORG.institute.logo} alt="IIT Delhi logo" className="logo-iitd" />
            </a>
          </div>
          <p className="dim">
            <a href={ORG.lab.url} target="_blank" rel="noreferrer">RISE Lab</a>,{' '}
            <a href={ORG.centre.url} target="_blank" rel="noreferrer">{ORG.centre.full} (CBME)</a>,{' '}
            <a href={ORG.institute.url} target="_blank" rel="noreferrer">{ORG.institute.full}</a>
          </p>
        </div>
        <div className="foot-links">
          <p>Project</p>
          <Link to="/analyze">Analyse a video</Link>
          <Link to="/guide">Recording guide</Link>
          <Link to="/markers">Marker CSV template</Link>
          <Link to="/cite">How to cite</Link>
          <a href={APP.repo} target="_blank" rel="noreferrer">Source code</a>
        </div>
        <div className="foot-links">
          <p>Legal</p>
          <Link to="/about#license">License: {APP.license}</Link>
          <Link to="/privacy">Privacy policy</Link>
          <Link to="/terms">Terms of use</Link>
          <Link to="/terms#disclaimer">Not a medical device</Link>
        </div>
      </div>
      <div className="foot-cite">
        <span className="dim">If you use {APP.name} in research, please cite:</span>
        <span className="mono cite-line">{CITATION.text}</span>
      </div>
      <div className="foot-bottom dim">
        <span>© {APP.year} {PEOPLE.developer.name}, RISE Lab, CBME, IIT Delhi · Released under the {APP.license}</span>
        <span>v{APP.version} · No videos are stored · No tracking</span>
      </div>
    </footer>
  )
}

export default function SiteLayout() {
  const loc = useLocation()
  useEffect(() => {
    if (loc.hash) {
      document.getElementById(loc.hash.slice(1))?.scrollIntoView({ behavior: 'smooth' })
    } else {
      window.scrollTo(0, 0)
    }
  }, [loc.pathname, loc.hash])
  return (
    <div className="site">
      <Header />
      <main className="site-main"><Outlet /></main>
      <Footer />
    </div>
  )
}
