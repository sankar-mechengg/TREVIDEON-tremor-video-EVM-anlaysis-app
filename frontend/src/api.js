// TREVIDEON API client.
// Local mode: the backend serves this page, so requests are same-origin.
// Hosted mode: the page (Vercel) talks to VITE_API_BASE (Railway); every request carries the per-tab session id.

export const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')
const SKEY = 'trevideon.session'

let session = null
try { session = sessionStorage.getItem(SKEY) } catch { /* private mode */ }

export const getSession = () => session
const setSession = (s) => {
  session = s
  try { s ? sessionStorage.setItem(SKEY, s) : sessionStorage.removeItem(SKEY) } catch { /* ignore */ }
}

async function req(path, opts = {}) {
  const headers = { ...(opts.raw ? {} : { 'Content-Type': 'application/json' }), ...(opts.headers || {}) }
  if (session) headers['X-Session'] = session
  const r = await fetch(API_BASE + path, {
    method: opts.method || 'GET',
    headers,
    body: opts.raw ? opts.body : opts.body ? JSON.stringify(opts.body) : undefined,
  })
  if (!r.ok) {
    let msg = r.statusText
    try {
      const det = (await r.json()).detail
      msg = Array.isArray(det) ? det.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ') : det || msg
    } catch { /* ignore */ }
    const e = new Error(msg)
    e.status = r.status
    throw e
  }
  const ct = r.headers.get('content-type') || ''
  return ct.includes('json') ? r.json() : r.text()
}

const enc = encodeURIComponent
const withS = (p) => (session ? `${p}${p.includes('?') ? '&' : '?'}s=${session}` : p)
export const url = (p) => (p.startsWith('http') ? p : API_BASE + withS(p))

export const api = {
  health: () => req('/api/health'),
  config: () => req('/api/config'),

  // hosted sessions
  async ensureSession(mode) {
    if (mode !== 'hosted') return null
    if (session) {
      try { await req('/api/session'); return session } catch { setSession(null) }
    }
    const r = await req('/api/session', { method: 'POST' })
    setSession(r.session)
    return r.session
  },
  sessionInfo: () => req('/api/session'),
  async deleteSession() {
    const r = await req('/api/session', { method: 'DELETE' })
    setSession(null)
    return r
  },
  endSessionBeacon() {
    if (!session) return
    try { navigator.sendBeacon(`${API_BASE}/api/session/end?s=${session}`) } catch { /* ignore */ }
  },

  // local library
  library: () => req('/api/library'),
  setLibrary: (data_dir, results_dir) => req('/api/library', { method: 'POST', body: { data_dir, results_dir } }),
  browse: (kind = 'data') => req(`/api/library/browse?kind=${kind}`, { method: 'POST' }),

  // uploads (hosted): 8 MB chunks; a lost chunk is retried, a 409 tells where to resume
  async upload(file, onProgress, signal) {
    const init = await req('/api/uploads', { method: 'POST', body: { filename: file.name, size: file.size } })
    let off = 0
    let tries = 0
    while (off < file.size) {
      if (signal?.aborted) throw new Error('Upload cancelled')
      const chunk = file.slice(off, off + init.chunk_size)
      let r
      try {
        r = await fetch(`${API_BASE}/api/uploads/${init.upload_id}?offset=${off}`,
          { method: 'PUT', body: chunk, headers: { 'X-Session': session || '' }, signal })
      } catch (e) {
        if (++tries > 5) throw e
        await new Promise((ok) => setTimeout(ok, 1000 * tries))
        continue
      }
      if (r.ok || r.status === 409) {
        off = (await r.json()).received
        tries = 0
        onProgress?.(off / file.size)
      } else {
        let msg = r.statusText
        try { msg = (await r.json()).detail || msg } catch { /* ignore */ }
        throw new Error(msg)
      }
    }
    return req(`/api/uploads/${init.upload_id}/complete`, { method: 'POST' })
  },
  deleteVideo: (v) => req(`/api/videos/${enc(v)}`, { method: 'DELETE' }),

  videos: () => req('/api/videos'),
  thumbUrl: (v) => url(`/api/videos/${enc(v)}/thumb`),
  rawUrl: (v) => url(`/api/videos/${enc(v)}/raw`),
  scan: (v, d = 20) => req(`/api/videos/${enc(v)}/scan?duration_s=${d}`),
  results: (v) => req(`/api/videos/${enc(v)}/results`),

  // markers
  markers: (v) => req(`/api/videos/${enc(v)}/markers`),
  uploadMarkers: (v, text) => req(`/api/videos/${enc(v)}/markers`, { method: 'POST', raw: true, body: text, headers: { 'Content-Type': 'text/csv' } }),
  deleteMarkers: (v) => req(`/api/videos/${enc(v)}/markers`, { method: 'DELETE' }),
  templateUrl: () => API_BASE + '/api/markers/template',

  // processing
  check: (video, settings) => req('/api/check', { method: 'POST', body: { video, settings } }),
  process: (video, settings) => req('/api/process', { method: 'POST', body: { video, settings } }),
  processAll: (items) => req('/api/process_all', { method: 'POST', body: { items } }),
  jobs: () => req('/api/jobs'),
  job: (id) => req(`/api/jobs/${id}`),
  cancel: (id) => req(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  result: (v, ka, kb) => req(`/api/result/${enc(v)}/${ka}${kb ? `?key_b=${kb}` : ''}`),

  // 3D kinematics
  k3dCheck: (video, settings) => req('/api/k3d/check', { method: 'POST', body: { video, settings } }),
  k3dProcess: (video, settings) => req('/api/k3d/process', { method: 'POST', body: { video, settings } }),
  k3dProcessAll: (items) => req('/api/k3d/process_all', { method: 'POST', body: { items } }),
  k3dResults: (v) => req(`/api/videos/${enc(v)}/k3d`),
  k3dResult: (v, k, kv) => req(`/api/k3d/${enc(v)}/${k}${kv ? `?key_v=${kv}` : ''}`),
  async fetchJson(path) {
    const r = await fetch(url(path))
    if (!r.ok) throw new Error(r.statusText)
    return r.json()
  },
  handFbxUrl: (side) => API_BASE + `/api/assets/hand/${side}.fbx`,
}

export const ANCHOR_COLORS = {
  thumb: '#ff6b6b', index: '#ffd93d', middle: '#6bcb77', ring: '#4d96ff', pinky: '#c77dff',
  dorsum: '#9aa4b2', hand: '#f1f5f9',
}

export const fmtTime = (s) => {
  if (!isFinite(s)) return '--:--'
  const m = Math.floor(s / 60)
  const r = s - m * 60
  return `${m}:${r.toFixed(2).padStart(5, '0')}`
}

export const fmtDur = (s) => {
  if (!isFinite(s)) return '-'
  const m = Math.floor(s / 60)
  const r = Math.round(s - m * 60)
  return m ? `${m}m ${String(r).padStart(2, '0')}s` : `${r}s`
}

export const fmtBytes = (b) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(0)} MB`)
