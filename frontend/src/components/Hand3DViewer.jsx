import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js'
import { api, url } from '../api.js'

/*
 * 3D view of the reconstructed hand.
 * World data: X right, Y forward, Z up, millimetres. three.js: Y up -> (x, y, z)_three = (X, Z, -Y) * 0.001 (metres).
 * The hand can follow its measured pose, a slow (low-passed) wrist pose, or a frozen mean wrist pose; finger
 * vibration (tremor band, model or hybrid) is added on top with a gain alpha (3D Lagrangian magnification).
 */
export const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
export const CHAIN = { thumb: [1, 2, 3, 4], index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20] }
export const FCOL = { thumb: '#ff6b6b', index: '#ffd93d', middle: '#6bcb77', ring: '#4d96ff', pinky: '#c77dff', wrist: '#9aa4b2' }
const RIGID = { thumb: [2, 4], index: [5, 8], middle: [9, 12], ring: [13, 16], pinky: [17, 20] }
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17]]
const S = 0.001
const AX = ['#ff5a5a', '#5ad15a', '#4c8dff']
const toT = (x, y, z, out = new THREE.Vector3()) => out.set(x * S, z * S, -y * S)
const dirT = (x, y, z, out = new THREE.Vector3()) => out.set(x, z, -y)

// ---------------------------------------------------------------------------------------------- pose maths
function matVec(R9, v) {
  return [R9[0] * v[0] + R9[1] * v[1] + R9[2] * v[2], R9[3] * v[0] + R9[4] * v[1] + R9[5] * v[2],
    R9[6] * v[0] + R9[7] * v[1] + R9[8] * v[2]]
}
function matTVec(R9, v) {
  return [R9[0] * v[0] + R9[3] * v[1] + R9[6] * v[2], R9[1] * v[0] + R9[4] * v[1] + R9[7] * v[2],
    R9[2] * v[0] + R9[5] * v[1] + R9[8] * v[2]]
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s]
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const norm = (a) => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n] }

/** Wrist pose (R row-major 3x3, origin mm) for frame i in the chosen wrist mode. */
function wristPose(d, i, mode) {
  if (mode === 'frozen') return { R: d.palm_mean.R, o: d.palm_mean.o }
  const src = mode === 'slow' ? d.palm_slow : d.palm
  return { R: src.R[i], o: src.o[i] }
}

/** 21 joints (world mm) for frame i. */
export function jointsAt(d, i, mode, alpha, source) {
  if (mode === 'follow' && alpha === 1) return d.joints_world[i]
  const { R, o } = wristPose(d, i, mode)
  const low = d.joints_palm_low[i]
  const band = (source === 'hybrid' ? d.band_hybrid : d.band_model)[i]
  const out = new Array(21)
  for (let j = 0; j < 21; j++) out[j] = add(matVec(R, add(low[j], scl(band[j], alpha))), o)
  return out
}

function palmBasis(J, right) {
  const y = norm(sub(J[9], J[0]))
  const a = sub(J[17], J[0]), b = sub(J[5], J[0])
  let z = right ? cross(a, b) : cross(b, a)
  z = norm(sub(z, scl(y, dot(z, y))))
  const x = cross(y, z)
  return [x, y, z]
}

// ---------------------------------------------------------------------------------------------- VibeSkin rig
const RIG_JOINT = { hand: 0, thumb1: 1, thumb2: 2, thumb3: 3, thumb_ignore: 4, index1: 5, index2: 6, index3: 7, index_ignore: 8,
  middle1: 9, middle2: 10, middle3: 11, middle_ignore: 12, ring1: 13, ring2: 14, ring3: 15, ring_ignore: 16,
  pinky1: 17, pinky2: 18, pinky3: 19, pinky_ignore: 20 }
const RIG_CHAINS = [['thumb1', 'thumb2', 'thumb3', 'thumb_ignore'], ['index1', 'index2', 'index3', 'index_ignore'],
  ['middle1', 'middle2', 'middle3', 'middle_ignore'], ['ring1', 'ring2', 'ring3', 'ring_ignore'],
  ['pinky1', 'pinky2', 'pinky3', 'pinky_ignore']]

function setupRig(fbx, right) {
  const bones = {}
  fbx.traverse((o) => {
    if (!o.isBone && o.type !== 'Bone' && o.type !== 'Object3D') return
    const m = o.name.match(/b_[rl]_(hand|thumb[0-3]|index[1-3]|middle[1-3]|ring[1-3]|pinky[0-3]|(?:thumb|index|middle|ring|pinky)_ignore)$/)
    if (m && !bones[m[1]]) bones[m[1]] = o
  })
  const need = ['hand', 'index1', 'middle1', 'pinky1', 'thumb1']
  if (!need.every((k) => bones[k])) return null
  fbx.updateMatrixWorld(true)
  const P = {}
  for (const [k, b] of Object.entries(bones)) P[k] = b.getWorldPosition(new THREE.Vector3())
  const arr = (v) => [v.x, v.y, v.z]
  const Jb = []
  Jb[0] = arr(P.hand); Jb[5] = arr(P.index1); Jb[9] = arr(P.middle1); Jb[17] = arr(P.pinky1)
  const [bx, by, bz] = palmBasis(Jb, right)
  const Rbind = new THREE.Matrix4().makeBasis(new THREE.Vector3(...bx), new THREE.Vector3(...by), new THREE.Vector3(...bz))
  const bind = {}
  for (const [k, b] of Object.entries(bones)) bind[k] = b.quaternion.clone()
  fbx.traverse((o) => {
    if (o.isMesh || o.isSkinnedMesh) {
      o.material = new THREE.MeshStandardMaterial({ color: 0xd9a58c, roughness: 0.55, metalness: 0.0, transparent: true, opacity: 0.92 })
      o.frustumCulled = false
      o.castShadow = false
    }
  })
  return { fbx, bones, bind, Rbind, handLen: P.hand.distanceTo(P.middle1), handBind: P.hand.clone() }
}

const _q = new THREE.Quaternion(), _qa = new THREE.Quaternion(), _qp = new THREE.Quaternion()
const _u = new THREE.Vector3(), _v = new THREE.Vector3(), _pa = new THREE.Vector3(), _pb = new THREE.Vector3()

function poseRig(rig, Jt, right) {
  // Jt: 21 THREE.Vector3 (three coords). Whole model: palm basis of the bind pose -> current palm basis.
  const arr = (v) => [v.x, v.y, v.z]
  const J = Jt.map(arr)
  const [x, y, z] = palmBasis(J, right)
  const Rnow = new THREE.Matrix4().makeBasis(new THREE.Vector3(...x), new THREE.Vector3(...y), new THREE.Vector3(...z))
  const Rm = Rnow.clone().multiply(rig.Rbind.clone().transpose())
  const s = Jt[0].distanceTo(Jt[9]) / (rig.handLen || 1)
  rig.fbx.quaternion.setFromRotationMatrix(Rm)
  rig.fbx.scale.setScalar(s)
  rig.fbx.position.set(0, 0, 0)
  for (const [k, q] of Object.entries(rig.bind)) rig.bones[k].quaternion.copy(q)
  rig.fbx.updateMatrixWorld(true)
  const h = rig.bones.hand.getWorldPosition(new THREE.Vector3())
  rig.fbx.position.copy(Jt[0]).sub(h)
  rig.fbx.updateMatrixWorld(true)
  for (const chain of RIG_CHAINS) {
    for (let k = 0; k < chain.length - 1; k++) {
      const b = rig.bones[chain[k]], c = rig.bones[chain[k + 1]]
      if (!b || !c) break
      b.getWorldPosition(_pa); c.getWorldPosition(_pb)
      _u.subVectors(_pb, _pa).normalize()
      _v.subVectors(Jt[RIG_JOINT[chain[k + 1]]], Jt[RIG_JOINT[chain[k]]]).normalize()
      if (!isFinite(_v.x)) continue
      _q.setFromUnitVectors(_u, _v)
      b.getWorldQuaternion(_qa)
      b.parent.getWorldQuaternion(_qp)
      b.quaternion.copy(_qp.invert().multiply(_q.multiply(_qa)))
      b.updateMatrixWorld(true)
    }
  }
}

// ---------------------------------------------------------------------------------------------- component
export default function Hand3DViewer({ data, meshUrl, time = 0, controls, hasVibeSkin, onInfo }) {
  const host = useRef(null)
  const st = useRef({})
  const [ready, setReady] = useState(false)
  const d = data?.display
  const fps = data?.fps || 30
  const right = data?.hand?.appears_right !== false
  const T = data?.n_frames || 0

  // ---- scene set-up (once per data set)
  useEffect(() => {
    if (!d || !host.current) return
    const el = host.current
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio))
    renderer.setClearColor(0x0b0f16)
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.005, 50)
    // start with the recording camera's "up" so that the first view matches the video (set before the controls,
    // which keep the up axis they were created with)
    const Rc0 = d.camera.R[0]
    const up0 = dirT(-Rc0[1], -Rc0[4], -Rc0[7]).normalize()
    if (isFinite(up0.x)) camera.up.copy(up0)
    const orbit = new OrbitControls(camera, renderer.domElement)
    orbit.enableDamping = true
    scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x202530, 1.1))
    const dl = new THREE.DirectionalLight(0xffffff, 1.4)
    dl.position.set(0.5, 1.2, 0.8)
    scene.add(dl)

    // hand centre and floor
    const J0 = d.joints_world
    let c = [0, 0, 0]
    for (const j of J0[0]) c = add(c, scl(j, 1 / 21))
    const centre = toT(...c)
    const floorZ = data.world?.floor_aligned ? 0 : Math.min(...J0[0].map((p) => p[2])) - 120
    const grid = new THREE.GridHelper(data.world?.floor_aligned ? 4 : 1, data.world?.floor_aligned ? 40 : 20, 0x2e3b58, 0x1a2334)
    grid.position.set(centre.x, floorZ * S, centre.z)
    scene.add(grid)

    // world axes (at the world origin, 150 mm)
    const worldAxes = new THREE.Group()
    for (let k = 0; k < 3; k++) {
      const e = [0, 0, 0]; e[k] = 150
      const g = new THREE.BufferGeometry().setFromPoints([toT(0, 0, 0), toT(...e)])
      worldAxes.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: AX[k] })))
    }
    scene.add(worldAxes)

    const lineTriad = (len, width = 1) => {
      const g = new THREE.Group()
      for (let k = 0; k < 3; k++) {
        const geo = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3))
        const ln = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: AX[k], linewidth: width }))
        ln.userData.len = len
        g.add(ln)
      }
      return g
    }
    const wristTriad = lineTriad(45)
    const cyl = (r, color) => new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 10), new THREE.MeshStandardMaterial({ color }))
    const wristArrows = new THREE.Group()
    for (let k = 0; k < 3; k++) { const m = cyl(0.0016, AX[k]); wristArrows.add(m) }
    scene.add(wristTriad, wristArrows)
    const fingerTriads = {}
    for (const f of FINGERS) { fingerTriads[f] = lineTriad(18); scene.add(fingerTriads[f]) }
    const headTriad = lineTriad(70)
    scene.add(headTriad)

    // procedural hand
    const skel = new THREE.Group()
    const jointMeshes = []
    const boneMeshes = []
    const skin = new THREE.MeshStandardMaterial({ color: 0xd9a58c, roughness: 0.6 })
    for (let j = 0; j < 21; j++) {
      const fin = FINGERS.find((f) => CHAIN[f].includes(j))
      const m = new THREE.Mesh(new THREE.SphereGeometry(j === 0 ? 0.009 : CHAIN[fin]?.[3] === j ? 0.0065 : 0.0075, 14, 10),
        new THREE.MeshStandardMaterial({ color: fin ? FCOL[fin] : 0xbfc7d5, roughness: 0.4 }))
      skel.add(m); jointMeshes.push(m)
    }
    for (const [a, b] of BONES) {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 12), skin)
      m.userData = { a, b, r: a === 0 || (b - a === 4) ? 0.0055 : 0.0068 }
      skel.add(m); boneMeshes.push(m)
    }
    const palmGeo = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(5 * 3 * 3), 3))
    const palm = new THREE.Mesh(palmGeo, new THREE.MeshStandardMaterial({ color: 0xd9a58c, roughness: 0.65, side: THREE.DoubleSide, transparent: true, opacity: 0.85 }))
    skel.add(palm)
    scene.add(skel)

    // trails + ellipsoids
    const NTR = Math.round(fps * 1.0)
    const trails = {}
    const ellipsoids = {}
    for (const f of FINGERS) {
      const g = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(NTR * 3), 3))
      trails[f] = new THREE.Line(g, new THREE.LineBasicMaterial({ color: FCOL[f], transparent: true, opacity: 0.95 }))
      scene.add(trails[f])
      const e = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshStandardMaterial({ color: FCOL[f], transparent: true, opacity: 0.28, depthWrite: false }))
      e.matrixAutoUpdate = false
      ellipsoids[f] = e
      scene.add(e)
    }

    // MANO mesh (local only)
    let mano = null
    if (meshUrl && data.mesh?.faces) {
      fetch(url(meshUrl)).then((r) => r.arrayBuffer()).then((buf) => {
        const h = new Uint16Array(buf)
        const V = new Float32Array(h.length)
        for (let k = 0; k < h.length; k++) V[k] = THREE.DataUtils.fromHalfFloat(h[k])
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(778 * 3), 3))
        geo.setIndex(data.mesh.faces.flat())
        const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xe0b19b, roughness: 0.5, transparent: true, opacity: 0.9, side: THREE.DoubleSide }))
        m.frustumCulled = false
        m.visible = false
        scene.add(m)
        mano = { mesh: m, V }
        st.current.mano = mano
        st.current.dirty = true
        onInfo?.({ mano: true })
      }).catch(() => onInfo?.({ mano: false }))
    }

    // VibeSkin hand (local only)
    if (hasVibeSkin) {
      new FBXLoader().load(api.handFbxUrl(right ? 'right' : 'left'), (fbx) => {
        const rig = setupRig(fbx, right)
        if (!rig) { onInfo?.({ vibeskin: false }); return }
        fbx.visible = false
        scene.add(fbx)
        st.current.rig = rig
        st.current.dirty = true
        onInfo?.({ vibeskin: true })
      }, undefined, () => onInfo?.({ vibeskin: false }))
    }

    // initial camera: from the recording camera (head) towards the hand, a little behind it
    const co = d.camera.o[0]
    const headPos = toT(...co)
    const dir = headPos.clone().sub(centre)
    const dist = Math.min(0.6, Math.max(0.28, dir.length() * 0.8))
    camera.position.copy(centre.clone().add(dir.normalize().multiplyScalar(dist)))
    orbit.target.copy(centre)
    camera.lookAt(centre)
    orbit.update()

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h, false)
      renderer.domElement.style.width = '100%'
      renderer.domElement.style.height = '100%'
      camera.aspect = w / Math.max(1, h)
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()
    let raf
    const loop = () => {
      orbit.update()
      if (st.current.dirty !== false || orbit.enableDamping) renderer.render(scene, camera)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    st.current = { ...st.current, renderer, scene, camera, orbit, worldAxes, wristTriad, wristArrows, fingerTriads, headTriad,
      skel, jointMeshes, boneMeshes, palm, trails, ellipsoids, NTR, centre, grid, dirty: true }
    setReady(true)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      orbit.dispose()
      renderer.dispose()
      scene.traverse((o) => { o.geometry?.dispose?.(); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose()) })
      el.removeChild(renderer.domElement)
      st.current = {}
      setReady(false)
    }
  }, [d, meshUrl, hasVibeSkin]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---- per-frame pose
  const frame = Math.max(0, Math.min(T - 1, Math.round(time * fps)))
  const ctl = controls
  useEffect(() => {
    const s = st.current
    if (!ready || !s.scene || !d) return
    const { mode, alpha, source, hand, show } = ctl
    const J = jointsAt(d, frame, mode, alpha, source)
    const Jt = J.map((p) => toT(...p))
    const { R: Rw, o: ow } = wristPose(d, frame, mode)
    const useMano = hand === 'mano' && s.mano
    const useRig = hand === 'vibeskin' && s.rig
    s.skel.visible = !useMano && !useRig
    if (s.rig) s.rig.fbx.visible = !!useRig
    if (s.mano) s.mano.mesh.visible = !!useMano

    // procedural skeleton
    if (s.skel.visible) {
      for (let j = 0; j < 21; j++) s.jointMeshes[j].position.copy(Jt[j])
      const up = new THREE.Vector3(0, 1, 0)
      for (const m of s.boneMeshes) {
        const a = Jt[m.userData.a], b = Jt[m.userData.b]
        const v = b.clone().sub(a)
        const len = v.length()
        m.position.copy(a).addScaledVector(v, 0.5)
        m.quaternion.setFromUnitVectors(up, v.normalize())
        m.scale.set(m.userData.r, len, m.userData.r)
      }
      const pa = s.palm.geometry.attributes.position
      const tri = [[0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17], [1, 2, 5]]
      tri.forEach((t, k) => t.forEach((j, q) => pa.setXYZ(k * 3 + q, Jt[j].x, Jt[j].y, Jt[j].z)))
      pa.needsUpdate = true
      s.palm.geometry.computeVertexNormals()
    }
    if (useRig) poseRig(s.rig, Jt, right)
    if (useMano) {
      const pos = s.mano.mesh.geometry.attributes.position
      const base = frame * 778 * 3
      const P = d.palm.R[frame], po = d.palm.o[frame]
      for (let k = 0; k < 778; k++) {
        let v = [s.mano.V[base + 3 * k] * 1000, s.mano.V[base + 3 * k + 1] * 1000, s.mano.V[base + 3 * k + 2] * 1000]
        if (mode !== 'follow') v = add(matVec(Rw, matTVec(P, sub(v, po))), ow)
        const t3 = toT(...v)
        pos.setXYZ(k, t3.x, t3.y, t3.z)
      }
      pos.needsUpdate = true
      s.mano.mesh.geometry.computeVertexNormals()
    }

    // frames
    const setTriad = (g, o, axes, len) => {
      g.children.forEach((ln, k) => {
        const a = toT(...o), e = toT(...add(o, scl(axes[k], len ?? ln.userData.len)))
        const p = ln.geometry.attributes.position
        p.setXYZ(0, a.x, a.y, a.z); p.setXYZ(1, e.x, e.y, e.z); p.needsUpdate = true
      })
    }
    const cols = (R9) => [[R9[0], R9[3], R9[6]], [R9[1], R9[4], R9[7]], [R9[2], R9[5], R9[8]]]
    const wAxes = cols(Rw)
    s.wristTriad.visible = show.wrist
    s.wristArrows.visible = show.wrist
    if (show.wrist) {
      setTriad(s.wristTriad, ow, wAxes)
      const up = new THREE.Vector3(0, 1, 0)
      s.wristArrows.children.forEach((m, k) => {
        const a = toT(...ow), b = toT(...add(ow, scl(wAxes[k], 45)))
        const v = b.clone().sub(a)
        m.position.copy(a).addScaledVector(v, 0.5)
        m.quaternion.setFromUnitVectors(up, v.clone().normalize())
        m.scale.set(1, v.length(), 1)
      })
    }
    const zp = wAxes[2]
    for (const f of FINGERS) {
      const g = s.fingerTriads[f]
      g.visible = show.fingers
      if (!show.fingers) continue
      const [m0, t0] = RIGID[f]
      const x = norm(sub(J[t0], J[m0]))
      const z = norm(sub(zp, scl(x, dot(zp, x))))
      const y = cross(z, x)
      setTriad(g, J[m0], [x, y, z])
    }
    s.worldAxes.visible = show.world
    s.grid.visible = show.world
    s.headTriad.visible = show.head && mode === 'follow'
    if (s.headTriad.visible) setTriad(s.headTriad, d.camera.o[frame], cols(d.camera.R[frame]))

    // trails (fingertip paths over the last second, same display transform)
    for (const f of FINGERS) {
      const tr = s.trails[f]
      tr.visible = show.trails
      if (!show.trails) continue
      const p = tr.geometry.attributes.position
      const tip = CHAIN[f][3]
      for (let k = 0; k < s.NTR; k++) {
        const i = Math.max(0, frame - (s.NTR - 1 - k))
        const Ji = jointsAt(d, i, mode, alpha, source)
        const v = toT(...Ji[tip])
        p.setXYZ(k, v.x, v.y, v.z)
      }
      p.needsUpdate = true
      tr.geometry.computeBoundingSphere()
    }
    // tremor ellipsoids (2 sigma, x alpha) at the fingertips, oriented with the displayed wrist frame
    const ell = data.ellipsoids?.[source] || {}
    for (const f of FINGERS) {
      const e = s.ellipsoids[f]
      const E = ell[f]
      e.visible = show.ellipsoids && !!E
      if (!e.visible) continue
      const tip = J[CHAIN[f][3]]
      const cols3 = E.dirs.map((dv, k) => { const w = matVec(Rw, dv); return dirT(...w).multiplyScalar(Math.max(E.axes[k] * alpha, 0.05) * S) })
      const pos = toT(...tip)
      e.matrix.makeBasis(cols3[0], cols3[1], cols3[2]).setPosition(pos)
    }
    if (ctl.followCam !== false) {
      const tp = mode === 'frozen' ? d.palm_mean : null
      const R9 = tp ? tp.R : d.palm_slow.R[frame]
      const o9 = tp ? tp.o : d.palm_slow.o[frame]
      let mc = [0, 0, 0]
      for (const p of d.joints_palm_low[frame]) mc = add(mc, scl(p, 1 / 21))
      const want = toT(...add(matVec(R9, mc), o9))
      const delta = want.clone().sub(s.orbit.target)
      s.camera.position.add(delta)
      s.orbit.target.copy(want)
    }
    s.dirty = true
  }, [ready, frame, ctl, d, data, right])

  return <div ref={host} className="viewer3d" />
}

/** Magnification that makes the median fingertip tremor ellipsoid (2 sigma) about `target_mm` long. */
export function autoAlpha(data, source = 'hybrid', target_mm = 8) {
  const E = data?.ellipsoids?.[source]
  const ax = E ? Object.values(E).map((e) => e.axes?.[0]).filter((v) => v > 0).sort((a, b) => a - b) : []
  if (!ax.length) return 50
  const med = ax[Math.floor(ax.length / 2)]
  const a = target_mm / med
  const nice = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 400]
  return nice.reduce((b, v) => (Math.abs(Math.log(v / a)) < Math.abs(Math.log(b / a)) ? v : b), 1)
}

export function useViewerControls(initial) {
  const [c, setC] = useState(() => ({ mode: 'follow', alpha: 1, source: 'hybrid', hand: 'skeleton', followCam: true,
    show: { world: true, wrist: true, fingers: true, head: false, trails: false, ellipsoids: false }, ...initial }))
  const api_ = useMemo(() => ({
    set: (k, v) => setC((s) => ({ ...s, [k]: v })),
    toggle: (k) => setC((s) => ({ ...s, show: { ...s.show, [k]: !s.show[k] } })),
  }), [])
  return [c, api_, setC]
}
