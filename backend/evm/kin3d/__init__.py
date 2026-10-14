"""3D hand kinematics: monocular 3D hand pose -> world / wrist (palm) / finger body frames -> finger vibration.

Modules
  tracking   MediaPipe dominant-hand tracking with 3D world landmarks
  wilor      WiLoR (CVPR 2025) MANO hand-mesh model on hand crops (GPU), test-time-augmented re-runs
  egomotion  head-mounted camera motion from the static background (floor-plane homography)
  frames     body frames, relative poses, finger angles, joint angles
  hybrid     sub-pixel optical-flow finger vibration expressed in the wrist frame
  signals    band-pass, spectra, spectrograms, coherence, ellipsoids, metrics
  exports    CSV / MATLAB / BVH / glTF / PNG figures
  pipeline   the cached job (stage K: kinematics, stage V: segmented display video)
"""
