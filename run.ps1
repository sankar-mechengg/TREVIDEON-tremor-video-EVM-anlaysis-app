<#
.SYNOPSIS
  Runs TREVIDEON (TRemor Eulerian VIDEO magnificatiON) on this computer and opens it in the browser.

.DESCRIPTION
  First run (a few minutes, needs internet):
    * creates backend\.venv with Python 3.12 (3.13 / 3.11 also work; MediaPipe needs <= 3.13)
    * installs PyTorch (CUDA 12.8 build when an NVIDIA GPU is found, otherwise the CPU build) and the
      Python packages in backend\requirements.txt
    * downloads the model files into backend\models (MediaPipe hand landmarker + multiclass segmenter,
      SAM 2.1 small for GPU hand segmentation)
    * with an NVIDIA GPU: the WiLoR 3D hand model for the 3D Kinematics tab (code into backend\third_party\WiLoR,
      2.6 GB checkpoint into backend\models). WiLoR also needs the MANO hand model, which you download yourself
      (free registration, research licence) from https://mano.is.tue.mpg.de and unzip into ..\data\mano_data
      (or backend\models\mano). Without it the 3D tab uses MediaPipe.
    * installs the website packages and builds the website
  Later runs start in a few seconds. Everything runs locally: no video leaves this computer.

  In the app, use "Browse..." to choose the folder with your videos. Results are written to a sibling
  "results" folder (for ...\data\raw that is ...\results) and reused as a cache.

.PARAMETER DataDir     Folder with the raw videos to open first (default: ..\data\raw, if it exists).
.PARAMETER ResultsDir  Results folder (default: the sibling "results" folder of DataDir).
.PARAMETER Port        Port of the local server (default 8000).
.PARAMETER Dev         Run the website with hot reload (http://localhost:5173) for development.
.PARAMETER Rebuild     Rebuild the website even if it looks up to date.
.PARAMETER Cpu         Install / use the CPU-only PyTorch build even if a GPU is present.
.PARAMETER NoBrowser   Do not open the browser.
.PARAMETER NoWilor     Do not install the WiLoR 3D hand model (the 3D tab then uses MediaPipe).

.EXAMPLE
  .\run.ps1
  .\run.ps1 -DataDir "D:\study\videos"
  powershell -ExecutionPolicy Bypass -File .\run.ps1        # if scripts are blocked
#>
param(
  [string]$DataDir = '',
  [string]$ResultsDir = '',
  [int]$Port = 8000,
  [switch]$Dev,
  [switch]$Rebuild,
  [switch]$Cpu,
  [switch]$NoBrowser,
  [switch]$NoWilor
)

# 'Continue': Windows PowerShell 5.1 turns stderr output of native tools (pip/npm warnings) into terminating
# errors under 'Stop'. Failures are detected through $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
$Root = $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Venv = Join-Path $Backend '.venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$Req = Join-Path $Backend 'requirements.txt'
$Models = Join-Path $Backend 'models'
$ModelFiles = [ordered]@{
  'hand_landmarker.task'             = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
  'selfie_multiclass_256x256.tflite' = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
  'sam2.1_hiera_small.pt'            = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'
}

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host '  TREVIDEON  ' -ForegroundColor Black -BackgroundColor Cyan -NoNewline
Write-Host '  TRemor Eulerian VIDEO magnificatiON - RISE Lab, CBME, IIT Delhi' -ForegroundColor Gray
Write-Host ''

# ---------------------------------------------------------------- GPU?
$hasGpu = $false
if (-not $Cpu) {
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($smi) { & nvidia-smi -L *> $null; $hasGpu = ($LASTEXITCODE -eq 0) }
}

# ---------------------------------------------------------------- Python environment
if (-not (Test-Path $Py)) {
  Step 'Creating the Python environment (backend\.venv)'
  $ver = $null
  foreach ($v in '3.12', '3.13', '3.11') {
    & py "-$v" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $ver = $v; break }
  }
  if (-not $ver) { Fail 'Python 3.11-3.13 is required (MediaPipe has no wheels for 3.14 yet). Install Python 3.12 from python.org.' }
  & py "-$ver" -m venv $Venv
  & $Py -m pip install --upgrade pip --quiet
}

& $Py -c "import torch" 2>$null
$torchOk = ($LASTEXITCODE -eq 0)
if (-not $torchOk) {
  if ($hasGpu) {
    Step 'Installing PyTorch with CUDA 12.8 (NVIDIA GPU found; ~3 GB download)'
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
  } else {
    Step 'Installing PyTorch (CPU build)'
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
  }
  if ($LASTEXITCODE -ne 0) { Fail 'PyTorch installation failed' }
}

$reqHash = (Get-FileHash $Req -Algorithm SHA1).Hash
$reqStamp = Join-Path $Venv '.requirements.sha1'
if (-not (Test-Path $reqStamp) -or (Get-Content $reqStamp -Raw).Trim() -ne $reqHash) {
  Step 'Installing Python packages'
  $env:SAM2_BUILD_CUDA = '0'
  & $Py -m pip install -r $Req --quiet
  if ($LASTEXITCODE -ne 0) { Fail 'pip install failed' }
  # mediapipe depends on opencv-contrib-python; a separately installed opencv-python would shadow it
  $pkgs = & $Py -m pip list --format=freeze 2>$null
  if ($pkgs -match '^opencv-python==') {
    & $Py -m pip uninstall -y opencv-python --quiet
    & $Py -m pip install --force-reinstall --no-deps opencv-contrib-python --quiet
  }
  Set-Content -Path $reqStamp -Value $reqHash
}

# ---------------------------------------------------------------- models
New-Item -ItemType Directory -Force $Models | Out-Null
foreach ($m in $ModelFiles.Keys) {
  $dst = Join-Path $Models $m
  if (-not (Test-Path $dst)) {
    if ($m -like '*.pt' -and -not $hasGpu) { continue }   # SAM 2 is only used with a GPU
    Step "Downloading model $m"
    try { Invoke-WebRequest -Uri $ModelFiles[$m] -OutFile $dst -UseBasicParsing -ErrorAction Stop }
    catch { Remove-Item $dst -ErrorAction SilentlyContinue; Fail "Could not download $m ($($_.Exception.Message))" }
  }
}

# ---------------------------------------------------------------- WiLoR (3D Kinematics; optional, GPU)
function Get-File([string]$url, [string]$dst) {
  $part = "$dst.part"
  $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
  if ($curl) { & curl.exe -L --fail --retry 5 -C - -o $part $url; $ok = ($LASTEXITCODE -eq 0) }
  else {
    $ProgressPreference = 'SilentlyContinue'
    try { Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing -ErrorAction Stop; $ok = $true } catch { $ok = $false }
  }
  if ($ok) { Move-Item -Force $part $dst }
  return $ok
}
if ($hasGpu -and -not $NoWilor) {
  $WilorReq = Join-Path $Backend 'requirements-wilor.txt'
  $wHash = (Get-FileHash $WilorReq -Algorithm SHA1).Hash
  $wStamp = Join-Path $Venv '.requirements-wilor.sha1'
  if (-not (Test-Path $wStamp) -or (Get-Content $wStamp -Raw).Trim() -ne $wHash) {
    Step 'Installing the WiLoR packages'
    & $Py -m pip install -r $WilorReq --quiet
    if ($LASTEXITCODE -eq 0) { Set-Content -Path $wStamp -Value $wHash }
  }
  $ThirdParty = Join-Path $Backend 'third_party'
  $WilorDir = Join-Path $ThirdParty 'WiLoR'
  if (-not (Test-Path (Join-Path $WilorDir 'wilor\models\wilor.py'))) {
    Step 'Downloading the WiLoR code (CC BY-NC-ND 4.0, github.com/rolpotamias/WiLoR)'
    New-Item -ItemType Directory -Force $ThirdParty | Out-Null
    $zip = Join-Path $ThirdParty 'wilor_main.zip'
    if (Get-File 'https://github.com/rolpotamias/WiLoR/archive/refs/heads/main.zip' $zip) {
      Expand-Archive -Path $zip -DestinationPath $ThirdParty -Force
      if (Test-Path $WilorDir) { Remove-Item -Recurse -Force $WilorDir }
      Rename-Item (Join-Path $ThirdParty 'WiLoR-main') 'WiLoR'
      Remove-Item $zip
    } else { Write-Host 'Could not download the WiLoR code; the 3D tab will use MediaPipe.' -ForegroundColor Yellow }
  }
  $WilorCkpt = Join-Path $Models 'wilor_final.ckpt'
  if (-not (Test-Path $WilorCkpt)) {
    Step 'Downloading the WiLoR checkpoint (2.6 GB, huggingface.co/spaces/rolpotamias/WiLoR)'
    if (-not (Get-File 'https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt' $WilorCkpt)) {
      Write-Host 'Could not download the WiLoR checkpoint; the 3D tab will use MediaPipe.' -ForegroundColor Yellow
    }
  }
  $manoDirs = @($env:TREVIDEON_MANO_DIR, (Join-Path $Models 'mano'), (Join-Path $Root '..\data\mano_data\models'),
                (Join-Path $Root '..\data\mano_data')) | Where-Object { $_ }
  if (-not ($manoDirs | Where-Object { Test-Path (Join-Path $_ 'MANO_RIGHT.pkl') })) {
    Write-Host ''
    Write-Host '  MANO hand model not found: the 3D tab uses MediaPipe until it is installed.' -ForegroundColor Yellow
    Write-Host '  Register at https://mano.is.tue.mpg.de, download "Models & Code" (mano_v1_2.zip) and unzip it so that' -ForegroundColor Gray
    Write-Host '  MANO_RIGHT.pkl is in ..\data\mano_data\models (or backend\models\mano). Research licence; not redistributed.' -ForegroundColor Gray
    Write-Host ''
  }
}

# ---------------------------------------------------------------- website
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Fail 'Node.js 18+ is required to build the website (https://nodejs.org).' }
Push-Location $Frontend
try {
  if (-not (Test-Path 'node_modules')) {
    Step 'Installing website packages'
    npm install --no-fund --no-audit
    if ($LASTEXITCODE -ne 0) { Fail 'npm install failed' }
  }
  if (-not $Dev) {
    $dist = Join-Path $Frontend 'dist\index.html'
    $needBuild = $Rebuild -or -not (Test-Path $dist)
    if (-not $needBuild) {
      $built = (Get-Item $dist).LastWriteTime
      $newer = Get-ChildItem -Recurse -File -Path 'src', 'public', 'index.html', 'vite.config.js', 'package.json' |
        Where-Object { $_.LastWriteTime -gt $built } | Select-Object -First 1
      $needBuild = [bool]$newer
    }
    if ($needBuild) {
      Step 'Building the website'
      npm run build
      if ($LASTEXITCODE -ne 0) { Fail 'Website build failed' }
    }
  }
} finally { Pop-Location }

# ---------------------------------------------------------------- start
$env:TREVIDEON_MODE = 'local'
if ($DataDir) {
  $resolved = Resolve-Path $DataDir -ErrorAction SilentlyContinue
  if (-not $resolved) { Fail "DataDir not found: $DataDir" }
  $env:TREVIDEON_DATA_DIR = $resolved.Path
}
if ($ResultsDir) { New-Item -ItemType Directory -Force $ResultsDir | Out-Null; $env:TREVIDEON_RESULTS_DIR = (Resolve-Path $ResultsDir).Path }

function Test-Up([string]$url) {
  try { Invoke-RestMethod -Uri $url -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

$api = "http://127.0.0.1:$Port"
$procs = @()
if (Test-Up "$api/api/health") {
  Write-Host "TREVIDEON is already running on $api" -ForegroundColor Yellow
} else {
  $busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if ($busy) {
    $start = $Port
    do { $Port++; $api = "http://127.0.0.1:$Port" } while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    Write-Host "Port $start is busy; using $api" -ForegroundColor Yellow
  }
  Step "Starting the TREVIDEON server on $api"
  $procs += Start-Process -FilePath $Py -ArgumentList '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', "$Port" `
    -WorkingDirectory $Backend -NoNewWindow -PassThru
  $ok = $false
  for ($i = 0; $i -lt 90; $i++) {
    if (Test-Up "$api/api/health") { $ok = $true; break }
    if ($procs[-1].HasExited) { Fail 'The server stopped during start-up (see the messages above).' }
    Start-Sleep -Milliseconds 500
  }
  if (-not $ok) { Fail 'The server did not start within 45 s.' }
}

$url = $api
if ($Dev) {
  Step 'Starting the website dev server on http://localhost:5173'
  $env:TREVIDEON_BACKEND = $api
  $procs += Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'npm run dev' -WorkingDirectory $Frontend -NoNewWindow -PassThru
  $url = 'http://localhost:5173'
  Start-Sleep -Seconds 3
}

Write-Host ''
Write-Host "  TREVIDEON is running at $url  ($(if ($hasGpu) { 'GPU' } else { 'CPU' }))" -ForegroundColor Green
Write-Host "  Open 'Analyse' and use 'Browse...' to choose your video folder." -ForegroundColor Gray
Write-Host '  Press Ctrl+C to stop.' -ForegroundColor DarkGray
Write-Host ''
if (-not $NoBrowser) { Start-Process "$url/analyze" }

try {
  if ($procs.Count -gt 0) { Wait-Process -Id ($procs | ForEach-Object Id) }
  else { while ($true) { Start-Sleep -Seconds 3600 } }
} finally {
  foreach ($p in $procs) {
    if (-not $p.HasExited) { & taskkill /PID $p.Id /T /F *> $null }
  }
}
