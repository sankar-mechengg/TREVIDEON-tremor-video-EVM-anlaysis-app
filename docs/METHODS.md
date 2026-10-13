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
