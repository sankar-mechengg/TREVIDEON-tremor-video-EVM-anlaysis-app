// Single source for names, links, credits and citation text used across the site.

export const APP = {
  name: 'TREVIDEON',
  expansion: 'TRemor Eulerian VIDEO magnificatiON',
  tagline: 'See the tremor your eyes cannot.',
  short: 'Open-source tremor identification from video with Eulerian Video Magnification',
  version: '1.0.0',
  year: 2026,
  license: 'Apache License 2.0',
  repo: 'https://github.com/sankar-mechengg/TREVIDEON',   // public repository (to be published)
}

export const PEOPLE = {
  developer: { name: 'Sankar Balasubramanian', linkedin: 'https://www.linkedin.com/in/sankar4' },
  pi: { name: 'Prof. Biswarup Mukherjee', title: 'Principal Investigator' },
}

export const ORG = {
  lab: { name: 'RISE Lab', full: 'Rehab, Instrumentation & Sensory Engineering Lab', url: 'https://riselab.iitd.ac.in/', logo: '/brand/rise-lab-logo.png' },
  centre: { name: 'CBME', full: 'Centre for Biomedical Engineering', url: 'https://cbme.iitd.ac.in/' },
  institute: { name: 'IIT Delhi', full: 'Indian Institute of Technology Delhi', url: 'https://home.iitd.ac.in/', logo: '/brand/iitd-logo.png' },
}

export const CITATION = {
  text: `Balasubramanian, S., & Mukherjee, B. (${APP.year}). ${APP.name}: ${APP.expansion} (Version ${APP.version}) [Computer software]. RISE Lab, Centre for Biomedical Engineering, Indian Institute of Technology Delhi. ${APP.repo}`,
  bibtex: `@software{trevideon${APP.year},
  author    = {Balasubramanian, Sankar and Mukherjee, Biswarup},
  title     = {{${APP.name}}: {${APP.expansion}}},
  year      = {${APP.year}},
  version   = {${APP.version}},
  publisher = {RISE Lab, Centre for Biomedical Engineering, Indian Institute of Technology Delhi},
  url       = {${APP.repo}},
  license   = {Apache-2.0},
  note      = {Open-source tool for tremor identification from video}
}`,
  doiNote: 'A Zenodo DOI and the companion journal article will be listed here once published.',
}

export const METHOD_REFS = [
  { key: 'Wu 2012', text: 'Wu H-Y, Rubinstein M, Shih E, Guttag J, Durand F, Freeman WT. Eulerian Video Magnification for Revealing Subtle Changes in the World. ACM Transactions on Graphics (SIGGRAPH) 31(4), 2012.', use: 'linear EVM' },
  { key: 'Wadhwa 2014', text: 'Wadhwa N, Rubinstein M, Durand F, Freeman WT. Riesz Pyramids for Fast Phase-Based Video Magnification. IEEE International Conference on Computational Photography (ICCP), 2014.', use: 'phase-based (Riesz) magnification' },
  { key: 'Kroeger 2016', text: 'Kroeger T, Timofte R, Dai D, Van Gool L. Fast Optical Flow using Dense Inverse Search. European Conference on Computer Vision (ECCV), 2016.', use: 'tremor measurement (optical flow)' },
  { key: 'Zhang 2020', text: 'Zhang F, Bazarevsky V, Vakunov A, Tkachenka A, Sung G, Chang C-L, Grundmann M. MediaPipe Hands: On-device Real-time Hand Tracking. CVPR Workshop on Computer Vision for AR/VR, 2020.', use: 'hand landmarks' },
  { key: 'Ravi 2024', text: 'Ravi N, Gabeur V, Hu Y-T, et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024.', use: 'hand segmentation (GPU)' },
]

export const SPECS = {
  minFps: 24,
  recFps: 100,
  maxUploadGB: 1,
  hostedSegment: 60,
  hostedHeight: 1080,
  ttlMin: 60,
}
