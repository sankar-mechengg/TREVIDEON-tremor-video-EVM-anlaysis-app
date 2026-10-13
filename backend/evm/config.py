"""TREVIDEON settings schema and recommended defaults.

Settings are split in stages so that the cache can be reused:
  * "process" settings -> hand isolation, stabilisation, magnification, motion fields, analysis (expensive)
  * "overlay" settings -> rendering of the tremor-envelope overlay video (cheap)
  * "runtime" settings -> how the work is executed (CPU / GPU); never part of a cache key
Changing only overlay settings re-renders the overlay from the cached motion fields.
"""

PIPELINE_VERSION = "1.3"  # bump to invalidate cached motion analysis (stage A)
OVERLAY_VERSION = "1.1"   # bump to invalidate only rendered overlays (stage B)

SCHEMA = [
    {
        "group": "Segment",
        "fields": [
            {"key": "start_s", "label": "Start time", "type": "number", "default": 5.0,
             "min": 0, "max": 100000, "step": 0.5, "unit": "s", "stage": "process",
             "help": "Where in the raw video the analysed segment starts."},
            {"key": "duration_s", "label": "Duration", "type": "number", "default": 20.0,
             "min": 0, "max": 100000, "step": 1, "unit": "s", "stage": "process",
             "help": "Length of the analysed segment. 0 = until the end of the video (slow for long videos)."},
            {"key": "work_height", "label": "Working resolution", "type": "select", "default": 720,
             "options": [[360, "360p (fast)"], [540, "540p"], [720, "720p (recommended)"], [1080, "1080p (slow)"],
                         [1440, "1440p (very slow)"], [0, "Native (no resizing)"]],
             "stage": "process",
             "help": "Frames are resized to this height before processing. Tremor amplitudes are reported in pixels at this resolution."},
            {"key": "repair_duplicates", "label": "Repair duplicated frames", "type": "bool", "default": True,
             "stage": "process",
             "help": "Screen / VR recordings and variable-frame-rate phone videos often repeat a frame at a regular "
                     "interval. The motion then stalls and jumps periodically, which creates a comb of false spectral "
                     "peaks (e.g. 2.5, 5, 10 Hz). Repeated frames are detected and replaced by a motion-interpolated "
                     "frame; the number repaired is reported."},
        ],
    },
    {
        "group": "Magnification",
        "fields": [
            {"key": "method", "label": "Method", "type": "select", "default": "riesz",
             "options": [["riesz", "Phase-based (Riesz pyramid)"], ["linear", "Linear EVM (Laplacian)"]],
             "stage": "process",
             "help": "Phase-based magnification (Wadhwa et al. 2014) amplifies motion with far less noise than classic linear EVM (Wu et al. 2012)."},
            {"key": "alpha", "label": "Amplification (α)", "type": "number", "default": 15,
             "min": 1, "max": 150, "step": 1, "unit": "×", "stage": "process",
             "help": "Motion amplification factor. 10–20 works well for physiological tremor at 720p; higher values exaggerate more but add artefacts."},
            {"key": "freq_lo", "label": "Band low cut", "type": "number", "default": 4.0,
             "min": 0.5, "max": 40, "step": 0.5, "unit": "Hz", "stage": "process", "fps_limit": True,
             "help": "Lower edge of the tremor band. 4 Hz rejects voluntary motion, postural sway and slow head motion."},
            {"key": "freq_hi", "label": "Band high cut", "type": "number", "default": 12.0,
             "min": 1, "max": 45, "step": 0.5, "unit": "Hz", "stage": "process", "fps_limit": True,
             "help": "Upper edge of the tremor band; it must stay below half the frame rate (15 Hz at 30 fps, 50 Hz at 100 fps). Physiological tremor peaks at 8–12 Hz."},
            {"key": "filter_order", "label": "Filter order", "type": "select", "default": 2,
             "options": [[1, "1 (soft)"], [2, "2 (recommended)"], [3, "3"], [4, "4 (sharp)"]], "stage": "process",
             "help": "Butterworth band-pass order."},
            {"key": "pyr_levels", "label": "Pyramid levels", "type": "number", "default": 0,
             "min": 0, "max": 7, "step": 1, "stage": "process",
             "help": "Number of spatial pyramid levels. 0 = automatic."},
            {"key": "phase_sigma", "label": "Phase smoothing σ", "type": "number", "default": 2.0,
             "min": 0, "max": 6, "step": 0.5, "unit": "px", "stage": "process",
             "help": "Amplitude-weighted spatial smoothing of the filtered phase (Riesz only). Reduces noise."},
            {"key": "motion_limit_px", "label": "Large-motion suppression", "type": "number", "default": 3.0,
             "min": 0, "max": 20, "step": 0.5, "unit": "px", "stage": "process",
             "help": "Riesz only: in-band motion larger than this (e.g. voluntary movement leaking into the band) is not amplified, which prevents ghosting. 0 = off."},
            {"key": "lambda_c", "label": "Cut-off wavelength λc", "type": "number", "default": 16,
             "min": 4, "max": 80, "step": 1, "unit": "px", "stage": "process",
             "help": "Linear EVM only: spatial wavelength below which amplification is attenuated to avoid artefacts."},
            {"key": "magnify_region", "label": "Magnify region", "type": "select", "default": "hand",
             "options": [["hand", "Hand only (soft mask)"], ["full", "Whole frame"]], "stage": "process",
             "help": "Restricting magnification to the hand avoids amplifying background noise and camera artefacts."},
        ],
    },
    {
        "group": "Hand & stabilisation",
        "fields": [
            {"key": "stabilization", "label": "Stabilisation", "type": "select", "default": "background",
             "options": [["background", "Background (remove camera motion)"],
                         ["hand", "Hand-anchored (finger vs. hand)"],
                         ["none", "None"]], "stage": "process",
             "help": "Background: camera shake is estimated from the scene (hand excluded) and removed. Hand-anchored: the back of the hand is held still so only finger motion relative to the hand remains."},
            {"key": "stab_cutoff_hz", "label": "Stabilisation cut-off", "type": "number", "default": 1.0,
             "min": 0.2, "max": 3.0, "step": 0.1, "unit": "Hz", "stage": "process",
             "help": "Motion above this frequency is removed by stabilisation; slower framing drift is kept."},
            {"key": "output_view", "label": "Output framing", "type": "select", "default": "original",
             "options": [["original", "Original camera view"], ["stabilized", "Stabilised view"]], "stage": "process",
             "help": "Original: magnified motion is mapped back into the original camera view so it aligns with the raw video."},
            {"key": "num_hands", "label": "Hands to track", "type": "select", "default": 2,
             "options": [[1, "1"], [2, "2"]], "stage": "process",
             "help": "Per-finger analysis uses the primary (largest, most persistent) hand; overlays cover all tracked hands."},
            {"key": "hand_confidence", "label": "Detection confidence", "type": "number", "default": 0.4,
             "min": 0.1, "max": 0.9, "step": 0.05, "stage": "process",
             "help": "MediaPipe hand detection/tracking confidence threshold."},
            {"key": "mask_margin", "label": "Hand mask margin", "type": "number", "default": 1.15,
             "min": 0.6, "max": 2.0, "step": 0.05, "unit": "×", "stage": "process",
             "help": "Scales the width of the hand mask built from the landmarks."},
        ],
    },
    {
        "group": "Overlay",
        "fields": [
            {"key": "overlay_base", "label": "Overlay drawn on", "type": "select", "default": "magnified",
             "options": [["magnified", "Magnified video"], ["original", "Original video"]], "stage": "overlay"},
            {"key": "envelope_window_s", "label": "Envelope window", "type": "number", "default": 0.4,
             "min": 0.1, "max": 3.0, "step": 0.1, "unit": "s", "stage": "overlay",
             "help": "Sliding RMS window used to compute the tremor envelope."},
            {"key": "colormap", "label": "Colour map", "type": "select", "default": "turbo",
             "options": [["turbo", "Turbo"], ["inferno", "Inferno"], ["plasma", "Plasma"], ["viridis", "Viridis"],
                         ["jet", "Jet"], ["hot", "Hot"]], "stage": "overlay"},
            {"key": "show_heatmap", "label": "Heatmap + iso-contours", "type": "bool", "default": True, "stage": "overlay"},
            {"key": "heatmap_opacity", "label": "Heatmap opacity", "type": "number", "default": 0.5,
             "min": 0.05, "max": 1.0, "step": 0.05, "stage": "overlay"},
            {"key": "contour_levels", "label": "Contour levels", "type": "number", "default": 5,
             "min": 0, "max": 12, "step": 1, "stage": "overlay"},
            {"key": "show_vectors", "label": "Micro-vector arrows", "type": "bool", "default": True, "stage": "overlay"},
            {"key": "vector_spacing", "label": "Arrow spacing", "type": "number", "default": 22,
             "min": 8, "max": 64, "step": 2, "unit": "px", "stage": "overlay"},
            {"key": "vector_scale", "label": "Arrow length", "type": "number", "default": 1.0,
             "min": 0.1, "max": 5.0, "step": 0.1, "unit": "×", "stage": "overlay",
             "help": "Arrow gain is auto-normalised so typical tremor fills the grid spacing; this multiplies it."},
            {"key": "show_halo", "label": "Hand halo envelope", "type": "bool", "default": True, "stage": "overlay"},
            {"key": "halo_width", "label": "Halo width", "type": "number", "default": 32,
             "min": 4, "max": 120, "step": 2, "unit": "px", "stage": "overlay"},
            {"key": "show_trails", "label": "Fingertip trails", "type": "bool", "default": True, "stage": "overlay"},
            {"key": "trail_seconds", "label": "Trail length", "type": "number", "default": 0.8,
             "min": 0.1, "max": 4.0, "step": 0.1, "unit": "s", "stage": "overlay"},
            {"key": "trail_scale", "label": "Trail gain", "type": "number", "default": 1.0,
             "min": 0.1, "max": 5.0, "step": 0.1, "unit": "×", "stage": "overlay"},
            {"key": "show_hud", "label": "HUD (labels, colour bar)", "type": "bool", "default": True, "stage": "overlay"},
        ],
    },
]

SCHEMA.insert(2, {
    "group": "Hand isolation (wrist-anchored)",
    "fields": [
        {"key": "hand_isolation", "label": "Isolate the dominant hand", "type": "bool", "default": False,
         "stage": "process",
         "help": "Segment the dominant hand, remove the background and pin the wrist centre (+) to a fixed point of a "
                 "square canvas before magnifying. Voluntary arm / hand movement is removed so that what remains is "
                 "tremor relative to the wrist. Produces a separate result next to the standard one."},
        {"key": "dominant_hand", "label": "Dominant hand", "type": "select", "default": "auto",
         "options": [["auto", "Auto (largest, most persistent)"], ["right", "Right"], ["left", "Left"]],
         "stage": "process", "requires": "hand_isolation",
         "help": "Which hand to isolate. Auto picks the largest hand that is visible most of the time."},
        {"key": "mirrored", "label": "Video is mirrored (selfie camera)", "type": "bool", "default": False,
         "stage": "process", "requires": "hand_isolation",
         "help": "Tick for front / selfie cameras that mirror the image. Egocentric (head / chest) cameras are not mirrored."},
        {"key": "segmenter", "label": "Hand segmentation", "type": "select", "default": "auto",
         "options": [["auto", "Auto (SAM 2 on GPU, MediaPipe on CPU)"], ["sam2", "SAM 2.1 (GPU)"],
                     ["mediapipe", "MediaPipe skin + landmarks (CPU)"], ["landmarks", "Landmark polygon (fastest)"]],
         "stage": "process", "requires": "hand_isolation",
         "help": "Model used to cut the hand out of the background. SAM 2.1 gives the cleanest edges but needs a GPU."},
        {"key": "anchor_cutoff_hz", "label": "Wrist pin cut-off", "type": "number", "default": 2.0,
         "min": 0, "max": 5, "step": 0.25, "unit": "Hz", "stage": "process", "requires": "hand_isolation",
         "help": "Wrist motion slower than this is removed (the + stays fixed); faster motion, including tremor of the "
                 "wrist itself, is kept. 0 = pin the wrist exactly in every frame (also removes wrist tremor)."},
        {"key": "hand_len_px", "label": "Normalised hand length", "type": "number", "default": 180,
         "min": 80, "max": 360, "step": 10, "unit": "px", "stage": "process", "requires": "hand_isolation",
         "help": "Every hand is scaled so that wrist -> middle-finger knuckle has this length, making amplitudes "
                 "comparable between people (also reported as % of hand length)."},
        {"key": "canvas_px", "label": "Canvas size", "type": "select", "default": 720,
         "options": [[512, "512 x 512"], [720, "720 x 720 (recommended)"], [960, "960 x 960"]],
         "stage": "process", "requires": "hand_isolation",
         "help": "Size of the square, wrist-anchored output video."},
    ],
})
SCHEMA.append({
    "group": "Compute",
    "fields": [
        {"key": "device", "label": "Magnification device", "type": "select", "default": "auto",
         "options": [["auto", "Auto (GPU if available)"], ["gpu", "GPU (CUDA)"], ["cpu", "CPU"]],
         "stage": "runtime",
         "help": "Where the Riesz-pyramid magnification runs. Does not change the measured tremor numbers "
                 "(those come from optical flow on the un-magnified video)."},
    ],
})

FIELDS = {f["key"]: f for g in SCHEMA for f in g["fields"]}
PROCESS_KEYS = [k for k, f in FIELDS.items() if f["stage"] == "process"]
OVERLAY_KEYS = [k for k, f in FIELDS.items() if f["stage"] == "overlay"]
RUNTIME_KEYS = [k for k, f in FIELDS.items() if f["stage"] == "runtime"]
DEFAULTS = {k: f["default"] for k, f in FIELDS.items()}
ISOLATION_KEYS = [k for k, f in FIELDS.items() if f.get("requires") == "hand_isolation"]


def max_band_hz(fps: float) -> float:
    """Highest usable band edge for a frame rate (a little below Nyquist)."""
    return round(0.45 * float(fps) * 2) / 2


def normalize(settings: dict | None, fps: float | None = None) -> dict:
    """Fill in defaults, coerce types and clamp to the schema ranges (and to the video's frame rate)."""
    out = dict(DEFAULTS)
    given = settings or {}
    # every field (defaults included) goes through the same coercion so cache keys are canonical
    for k, f in FIELDS.items():
        v = given.get(k)
        if v is None:
            v = f["default"]
        try:
            if f["type"] == "number":
                v = float(v)
                v = min(max(v, f["min"]), f["max"])
                if float(f.get("step", 1)).is_integer() and float(f["default"]).is_integer():
                    v = int(round(v))
                else:
                    v = round(v, 6)
            elif f["type"] == "bool":
                v = bool(v)
            elif f["type"] == "select":
                allowed = [o[0] for o in f["options"]]
                if v not in allowed:
                    # tolerate "720" vs 720
                    match = [a for a in allowed if str(a) == str(v)]
                    if not match:
                        continue
                    v = match[0]
        except (TypeError, ValueError):
            continue
        out[k] = v
    if fps:
        top = max_band_hz(fps)
        out["freq_hi"] = min(out["freq_hi"], top)
        out["freq_lo"] = min(out["freq_lo"], top - 1.0)
    if out["freq_hi"] <= out["freq_lo"] + 0.5:
        out["freq_hi"] = min(out["freq_lo"] + 1.0, FIELDS["freq_hi"]["max"])
    if not out["hand_isolation"]:
        # isolation sub-settings must not change the cache key of standard runs
        for k in ISOLATION_KEYS:
            out[k] = DEFAULTS[k]
    return out


def split(settings: dict, fps: float | None = None) -> tuple[dict, dict]:
    s = normalize(settings, fps)
    P = {k: s[k] for k in PROCESS_KEYS}
    if not P["repair_duplicates"]:
        P.pop("repair_duplicates")      # absent = no repair (keys of results made before this option)
    if not P["hand_isolation"]:
        # standard runs keep the exact process dict (and cache key) of earlier versions
        for k in ["hand_isolation"] + ISOLATION_KEYS:
            P.pop(k, None)
    return P, {k: s[k] for k in OVERLAY_KEYS}


def runtime(settings: dict | None) -> dict:
    s = normalize(settings)
    return {k: s[k] for k in RUNTIME_KEYS}
