import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import * as THREE from 'three'
import gsap from 'gsap'

// Canonical right hand (palm towards the viewer, fingers up), MediaPipe landmark order, hand length ~1.
const LM = [
  [0.0, -1.0], [-0.28, -0.85], [-0.48, -0.62], [-0.62, -0.42], [-0.74, -0.24],
  [-0.25, -0.25], [-0.29, 0.1], [-0.31, 0.32], [-0.32, 0.52],
  [-0.03, -0.22], [-0.03, 0.18], [-0.03, 0.43], [-0.03, 0.66],
  [0.18, -0.25], [0.21, 0.1], [0.23, 0.33], [0.24, 0.53],
  [0.37, -0.32], [0.43, -0.07], [0.47, 0.1], [0.5, 0.27],
]
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17]]
const PALM = [0, 1, 2, 5, 9, 13, 17]

function inPoly(x, y, poly) {
  let ins = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) ins = !ins
  }
  return ins
}

function buildHand(n = 7000) {
  const pos = new Float32Array(n * 3)
  const scatter = new Float32Array(n * 3)
  const rnd = new Float32Array(n)
  const weight = new Float32Array(n)
  const palm = PALM.map((i) => LM[i])
  const rand = (a, b) => a + Math.random() * (b - a)
  for (let k = 0; k < n; k++) {
    let x, y
    const r = Math.random()
    if (r < 0.62) {                                   // fingers / bones
      const [a, b] = BONES[Math.floor(Math.random() * BONES.length)]
      const t = Math.random()
      const [ax, ay] = LM[a]
      const [bx, by] = LM[b]
      const dx = bx - ax
      const dy = by - ay
      const L = Math.hypot(dx, dy) || 1
      const w = (b === 4 || b === 8 || b === 12 || b === 16 || b === 20 ? 0.055 : 0.075) * (1 - 0.25 * t)
      const o = (Math.random() * 2 - 1) * w
      x = ax + dx * t + (-dy / L) * o
      y = ay + dy * t + (dx / L) * o
    } else if (r < 0.92) {                            // palm fill
      do { x = rand(-0.75, 0.55); y = rand(-1.0, -0.15) } while (!inPoly(x, y, palm))
    } else {                                          // wrist / forearm
      x = rand(-0.2, 0.2) + 0.03 * Math.sin(k)
      y = rand(-1.45, -0.95)
    }
    pos[k * 3] = x
    pos[k * 3 + 1] = y
    pos[k * 3 + 2] = rand(-0.04, 0.04)
    const th = Math.random() * Math.PI * 2
    const rr = rand(2.5, 6)
    scatter[k * 3] = Math.cos(th) * rr
    scatter[k * 3 + 1] = Math.sin(th) * rr * 0.7
    scatter[k * 3 + 2] = rand(-4, 2)
    rnd[k] = Math.random()
    weight[k] = Math.min(1, Math.max(0.05, (y + 1.1) / 1.7)) // fingertips tremble most
  }
  return { pos, scatter, rnd, weight }
}

const VERT = /* glsl */ `
  attribute vec3 aScatter; attribute float aRnd; attribute float aWeight;
  uniform float uTime, uAssemble, uLensR, uMag, uPix, uSize;
  uniform vec2 uLens;
  varying float vHot; varying float vRnd;
  void main() {
    float a = smoothstep(0.0, 1.0, clamp(uAssemble * 1.6 - aRnd * 0.6, 0.0, 1.0));
    vec3 p = mix(aScatter, position, a);
    float d = distance(p.xy, uLens);
    float inside = 1.0 - smoothstep(uLensR * 0.82, uLensR, d);
    float f = 9.5;                                           // physiological tremor ~8-12 Hz
    vec2 tremor = vec2(sin(6.2832 * f * uTime + aRnd * 6.2832),
                       cos(6.2832 * f * 1.07 * uTime + aRnd * 3.1));
    float amp = 0.0022 * aWeight * mix(1.0, uMag, inside);
    p.xy += tremor * amp + (1.0 - a) * 0.0;
    vHot = inside * (0.55 + 0.45 * abs(tremor.x));
    vRnd = aRnd;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * uPix * (1.0 + inside * 0.7) / -mv.z;
  }`

const FRAG = /* glsl */ `
  varying float vHot; varying float vRnd;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c);
    if (r > 0.5) discard;
    float soft = smoothstep(0.5, 0.0, r);
    vec3 cool = mix(vec3(0.13, 0.83, 0.93), vec3(0.65, 0.55, 0.98), vRnd);
    vec3 hot = mix(vec3(1.0, 0.36, 0.55), vec3(1.0, 0.75, 0.25), vRnd);
    gl_FragColor = vec4(mix(cool, hot, vHot), soft * (0.42 + 0.5 * vHot));
  }`

function makeRing(segments = 160) {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segments * 3), 3))
  const m = new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0, blending: THREE.AdditiveBlending })
  const line = new THREE.LineLoop(g, m)
  line.userData = { segments, born: -10 }
  return line
}

const Hero3D = forwardRef(function Hero3D({ className = '' }, ref) {
  const mount = useRef(null)
  const api = useRef({})

  useImperativeHandle(ref, () => ({ launch: (done) => api.current.launch?.(done) ?? done?.() }), [])

  useEffect(() => {
    const el = mount.current
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches || new URLSearchParams(window.location.search).has('still')
    let renderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' })
    } catch {
      el.classList.add('no-webgl')            // no WebGL (old GPU / blocked): keep the static gradient background
      api.current.launch = (done) => done?.()
      return undefined
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100)
    camera.position.set(0.15, -0.25, 3.6)

    const hand = new THREE.Group()
    hand.position.set(0.9, 0.12, 0)
    scene.add(hand)
    const H = buildHand(window.innerWidth < 700 ? 4200 : 7500)
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(H.pos, 3))
    geo.setAttribute('aScatter', new THREE.BufferAttribute(H.scatter, 3))
    geo.setAttribute('aRnd', new THREE.BufferAttribute(H.rnd, 1))
    geo.setAttribute('aWeight', new THREE.BufferAttribute(H.weight, 1))
    const U = {
      uTime: { value: 0 }, uAssemble: { value: reduced ? 1 : 0 }, uLens: { value: new THREE.Vector2(-0.1, 0.2) },
      uLensR: { value: 0.32 }, uMag: { value: 14 }, uPix: { value: renderer.getPixelRatio() }, uSize: { value: 24 },
    }
    const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: U, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending })
    hand.add(new THREE.Points(geo, mat))

    // magnifier lens: ring + handle
    const lens = new THREE.Group()
    const lensRing = new THREE.Mesh(new THREE.RingGeometry(0.97, 1.0, 96),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.85, side: THREE.DoubleSide }))
    const lensGlow = new THREE.Mesh(new THREE.CircleGeometry(1, 64),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.05 }))
    const handle = new THREE.Mesh(new THREE.PlaneGeometry(0.09, 0.9),
      new THREE.MeshBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.8 }))
    handle.position.set(0.95, -0.95, 0)
    handle.rotation.z = Math.PI / 4
    lens.add(lensGlow, lensRing, handle)
    lens.position.z = 0.06
    hand.add(lens)

    // waveform rings rippling out of the lens
    const rings = Array.from({ length: 6 }, () => makeRing())
    rings.forEach((r) => hand.add(r))
    let nextRing = 0

    // dust
    const dustN = 900
    const dg = new THREE.BufferGeometry()
    const dp = new Float32Array(dustN * 3)
    for (let i = 0; i < dustN; i++) {
      dp[i * 3] = (Math.random() - 0.5) * 14
      dp[i * 3 + 1] = (Math.random() - 0.5) * 8
      dp[i * 3 + 2] = -Math.random() * 8 - 1
    }
    dg.setAttribute('position', new THREE.BufferAttribute(dp, 3))
    const dust = new THREE.Points(dg, new THREE.PointsMaterial({ color: 0x5b6b8c, size: 0.02, transparent: true, opacity: 0.6 }))
    scene.add(dust)

    // portal torus (used by launch)
    const portal = new THREE.Mesh(new THREE.TorusGeometry(1, 0.035, 24, 160),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0 }))
    portal.position.copy(hand.position)
    portal.position.z = 0.5
    portal.scale.setScalar(0.01)
    scene.add(portal)

    const resize = () => {
      const w = el.clientWidth
      const h = el.clientHeight
      renderer.setSize(w, h, false)
      camera.aspect = w / Math.max(h, 1)
      hand.position.x = w < 900 ? 0 : 0.95
      hand.position.y = w < 900 ? 0.35 : 0.12
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(el)

    // pointer: lens follows the mouse over the hand; hand tilts with the pointer
    const mouse = { x: 0, y: 0, active: false, last: 0 }
    const ray = new THREE.Raycaster()
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)
    const onMove = (e) => {
      const r = el.getBoundingClientRect()
      mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1
      mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1
      mouse.active = true
      mouse.last = performance.now()
    }
    window.addEventListener('pointermove', onMove)

    const clock = new THREE.Timer()
    const lensTarget = new THREE.Vector2(-0.1, 0.2)
    let raf
    let running = true
    const tmp = new THREE.Vector3()
    const loop = () => {
      if (!running) return
      raf = requestAnimationFrame(loop)
      clock.update()
      const t = clock.getElapsed()
      U.uTime.value = reduced ? 0 : t
      if (mouse.active && performance.now() - mouse.last < 2500) {
        ray.setFromCamera(new THREE.Vector2(mouse.x, mouse.y), camera)
        if (ray.ray.intersectPlane(plane, tmp)) {
          hand.worldToLocal(tmp)
          if (Math.abs(tmp.x) < 1.1 && tmp.y > -1.5 && tmp.y < 0.9) lensTarget.set(tmp.x, tmp.y)
        }
      } else {
        lensTarget.set(-0.1 + 0.42 * Math.sin(t * 0.35), -0.05 + 0.48 * Math.sin(t * 0.23 + 1.2))
      }
      U.uLens.value.lerp(lensTarget, 0.06)
      lens.position.x = U.uLens.value.x
      lens.position.y = U.uLens.value.y
      lens.scale.setScalar(U.uLensR.value)
      hand.rotation.y += ((mouse.active ? mouse.x * 0.25 : Math.sin(t * 0.2) * 0.12) - hand.rotation.y) * 0.04
      hand.rotation.x += ((mouse.active ? -mouse.y * 0.12 : 0) - hand.rotation.x) * 0.04
      dust.rotation.y = t * 0.01

      if (!reduced && t - (rings[nextRing].userData.born) > 0.6 * rings.length) {
        rings[nextRing].userData.born = t
        rings[nextRing].userData.cx = U.uLens.value.x
        rings[nextRing].userData.cy = U.uLens.value.y
        nextRing = (nextRing + 1) % rings.length
      }
      if (!reduced) {
        rings.forEach((ring) => {
          const age = t - ring.userData.born
          const life = 3.6
          const m = ring.material
          if (age > life || age < 0) { m.opacity = 0; return }
          const R = U.uLensR.value * (1 + age * 0.8)
          const arr = ring.geometry.attributes.position.array
          const S = ring.userData.segments
          for (let i = 0; i < S; i++) {
            const th = (i / S) * Math.PI * 2
            const wob = 0.018 * Math.sin(th * 12 + t * 9.5 * 2 * Math.PI * 0.1) * (1 - age / life)
            arr[i * 3] = ring.userData.cx + Math.cos(th) * (R + wob)
            arr[i * 3 + 1] = ring.userData.cy + Math.sin(th) * (R + wob)
            arr[i * 3 + 2] = 0.04
          }
          ring.geometry.attributes.position.needsUpdate = true
          m.opacity = 0.5 * (1 - age / life)
        })
      }
      renderer.render(scene, camera)
    }
    loop()

    let intro
    if (!reduced) {
      intro = gsap.timeline()
      intro.to(U.uAssemble, { value: 1, duration: 3.2, ease: 'power3.out' })
        .fromTo(lens.children[1].material, { opacity: 0 }, { opacity: 0.85, duration: 0.8 }, '-=1.2')
    }

    api.current.launch = (done) => {
      if (reduced) { done?.(); return }
      gsap.timeline({ onComplete: () => done?.() })
        .to(U.uLensR, { value: 1.6, duration: 0.7, ease: 'power2.in' }, 0)
        .to(U.uMag, { value: 40, duration: 0.7 }, 0)
        .to(portal.material, { opacity: 1, duration: 0.25 }, 0.1)
        .to(portal.scale, { x: 3.4, y: 3.4, z: 3.4, duration: 1.0, ease: 'power3.in' }, 0.1)
        .to(portal.rotation, { z: Math.PI, duration: 1.0 }, 0.1)
        .to(camera.position, { x: hand.position.x, y: hand.position.y, z: 0.35, duration: 1.05, ease: 'power3.in' }, 0.25)
        .to(el.parentElement?.querySelector('.portal-flash') || {}, { opacity: 1, duration: 0.35 }, 0.95)
    }

    return () => {
      running = false
      cancelAnimationFrame(raf)
      intro?.kill()
      ro.disconnect()
      window.removeEventListener('pointermove', onMove)
      scene.traverse((o) => {
        o.geometry?.dispose?.()
        if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose())
      })
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={mount} className={`hero3d ${className}`} aria-hidden="true" />
})

export default Hero3D
