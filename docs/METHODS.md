# TREVIDEON methods

1. **Segment and frame timing.** The chosen segment is read at the working resolution (720p by default).
   Repeated frames (frame difference < 15 % of the running median) are replaced by a motion-interpolated
   midpoint of their neighbours (bidirectional DIS flow, half-step warps); frames that jump about twice the
   usual amount without a preceding repeat are reported as suspected dropped frames.
2. **Hand tracking.** MediaPipe Hands gives 21 landmarks per hand per frame; tracks are gap-filled and
   low-passed at 5 Hz (removes detector jitter).
3. **Stabilisation.** Camera motion is estimated from background features (LK tracking, similarity RANSAC)
   and everything faster than 1 Hz is removed. Alternatively the back of the hand can be held still.
4. **Hand isolation (optional).** The dominant hand is segmented (SAM 2.1 prompted by the landmarks on a GPU;
   MediaPipe multiclass "body-skin" restricted to the landmark region on CPU; both united with a thin
   landmark polygon so that no finger is lost), the background removed, the image scaled so that
   wrist → middle-finger MCP = 180 px, and the wrist translated so that its slow path (< 2 Hz) stays on a
   fixed point (+). Orientation is not normalised, so rotational tremor about the wrist is kept. The
   standard measurement then runs on this canvas video without further stabilisation.
5. **Measurement.** Dense optical flow (DIS) between consecutive un-magnified frames, accumulated to
   displacement and band-passed (Butterworth, default 4–12 Hz, capped below 0.45 × fps). Eulerian field on
   an 8 × 8 px grid (heatmap, whole-hand envelope) and Lagrangian anchors on the five fingertip pads and the
   dorsum (spectra, per-finger metrics).
6. **Metrics.** Welch PSD (4 s windows) of the 1 Hz high-passed displacement; dominant frequency = peak of
   the velocity spectrum inside the band; RMS / P95 of the band-passed displacement; band power = in-band /
   > 1 Hz power; % of hand size; 0.5 s sliding-RMS envelopes; spectrograms (2 s windows).
7. **Magnification (visual only).** Phase-based Riesz-pyramid EVM (Wadhwa et al. 2014) with a streaming
   Butterworth band-pass, amplitude-weighted phase smoothing, large-motion suppression and phase clipping;
   or linear EVM (Wu et al. 2012). Runs on the GPU (PyTorch) when available.
8. **Markers.** For every marker segment (and a ±1 s window around every event) inside the analysed part:
   fingertip RMS, % of hand length, mean whole-hand envelope and dominant frequency.

## 3D kinematics (3D tab)

9. **Tracking and handedness.** MediaPipe follows the dominant hand (2D landmarks and metric 3D world
   landmarks). Whether it is the right or left hand comes from the image positions of the two hands (in an
   egocentric view the right hand is the right-most one; in a third-person view the left-most one), because
   MediaPipe's handedness classifier is unreliable for the back of the hand seen from a head-mounted camera;
   with one hand visible, the direction of the forearm decides (the right forearm runs towards the right of an
   egocentric image), and MediaPipe's label is the last resort. The side can also be set by hand.
10. **Camera (head) motion.** Background features (both hands masked out) are tracked with KLT relative to
   keyframes and a RANSAC homography maps first-frame to frame-t pixels. The head rotation is the polar part
   of K⁻¹HK with a pinhole camera from the given horizontal field of view; it is low-passed at 2 Hz (in-band
   background estimates are dominated by tracking noise) and removed. Head translation is not removed (with
   an assumed focal length, a focal-length error turns rotation into a spurious translation). The floor normal,
   from the plane-induced homographies (H = K(R + t nᵀ/d)K⁻¹ with n shared by all frames), turns the world
   axes upright (Z up) when it is well determined. Finger motion relative to the wrist does not depend on it.
11. **3D hand pose.** WiLoR (Potamias et al., CVPR 2025; MANO hand model) on a 256 px crop that follows the
   smoothed hand path (2 Hz) with one fixed size per clip (twice the landmark extent), so the crop itself does
   not jitter; left hands are mirrored into the crop and back. WiLoR's near-orthographic camera (5000 px focal
   length per 256 px) is converted to the real pinhole camera: depth from the apparent hand size, the hand
   rotated onto its viewing ray (perspective correction). MediaPipe's world landmarks are placed the same way
   and run alongside for comparison (and as the CPU / no-MANO fallback). An optional measured hand length
   rescales everything about the camera centre, which leaves the image projection unchanged.
12. **Coordinate frames.**
   *World* - first-frame camera with head rotation removed (Z up from the floor when available).
   *Wrist (palm)* - the wrist joint and the four finger knuckles (MCPs) fitted with Kabsch to a per-clip palm
   template; origin = wrist, y = towards the middle knuckle, z = back of the hand, x = y × z.
   *Rigid finger* - origin = knuckle (thumb: its MCP), x = knuckle → fingertip, z = palm normal made
   perpendicular to x, y = z × x. The finger's pose relative to the wrist, T_wrist⁻¹ · T_finger, is
   independent of the world frame and of any camera motion. Angles: flexion (out of the palm plane, + towards
   the palm), abduction (in the palm plane, + towards the thumb), twist (WiLoR's MANO rotation of the proximal
   phalanx, swing-twist decomposition). Joint flexion angles from consecutive bone vectors.
13. **Finger vibration: model vs hybrid.** *Model*: fingertip position / angles in the wrist frame directly from
   the 3D model. *Hybrid*: dense optical flow (DIS) on the un-magnified frames; the palm's rigid 2D motion
   (least-squares similarity fit to flow at eight points distal to the knuckles, avoiding the wrist-worn
   device) is subtracted from the fingertip-pad flow, giving each finger's own motion relative to the wrist
   in pixels; per-frame increments are scaled to millimetres with the model's metric depth and rotated into
   the low-passed (3 Hz) wrist frame, then accumulated. The flow measures motion across the image only; the
   visibility of each wrist axis (1 = across the image, 0 = along the line of sight) is reported.
14. **Noise floors.** *Model*: the 3D model is re-run on four crops shifted by 3 % of the crop size; half the
   PSD of the difference to the main run estimates its jitter (a lower bound: the re-runs see the same image).
   *Hybrid*: the residual of the rigid palm fit gives the per-frame flow noise σ² (skin texture at the hand's
   depth; × 1.5 for the prediction at the fingertip); accumulated white increments have the PSD
   2σ² / (fs · 4 sin²(πf/fs)). SNR = in-band signal power / in-band noise power.
15. **Signals and statistics.** For every series: band-pass (Butterworth, zero phase, default 4–12 Hz),
   velocity and acceleration of the band-passed signal (central differences), Welch PSD (4 s windows), band
   RMS, peak frequency, band power %, spectrogram of the principal tremor axis (2 s windows), magnitude-squared
   coherence between bodies (wrist position, wrist rotation, five fingers) averaged over the band, 2σ tremor
   ellipsoids of the fingertips in the wrist frame.
16. **Display and exports.** The 3D view adds the band-passed finger motion × α to the low-passed (3 Hz) pose in
   the wrist frame and shows it on the measured, slow or frozen (mean) wrist pose (3D Lagrangian
   magnification). Exports: CSV (world transforms, fingers in the wrist frame, joints, all series, joint
   angles, metrics, coherence), MATLAB .mat, BVH and glTF animation (Y-up; swing-only joint rotations of the
   clip-median skeleton), PNG figures at 300 dpi.
