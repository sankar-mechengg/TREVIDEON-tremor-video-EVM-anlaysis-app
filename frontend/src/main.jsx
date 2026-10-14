import { lazy, StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import SiteLayout from './site/Layout.jsx'
import Home from './site/Home.jsx'
import { About, Cite, Guide, Markers, NotFound, Privacy, Terms } from './site/Pages.jsx'
import './styles.css'
import './site/site.css'

const Analyzer = lazy(() => import('./app/Analyzer.jsx'))
const Kinematics = lazy(() => import('./app/Kinematics.jsx'))

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/analyze" element={<Suspense fallback={<div className="fatal">Loading…</div>}><Analyzer /></Suspense>} />
        <Route path="/analyze/3d" element={<Suspense fallback={<div className="fatal">Loading…</div>}><Kinematics /></Suspense>} />
        <Route element={<SiteLayout />}>
          <Route index element={<Home />} />
          <Route path="guide" element={<Guide />} />
          <Route path="markers" element={<Markers />} />
          <Route path="cite" element={<Cite />} />
          <Route path="privacy" element={<Privacy />} />
          <Route path="terms" element={<Terms />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
