# Clone joonson/syncnet_python, download weights, apply CPU patch (Windows).
# Run from repo root: .\scripts\setup_syncnet.ps1
param(
    [string]$RepoUrl = "https://github.com/joonson/syncnet_python.git"
)

$ErrorActionPreference = "Stop"
# scripts/ -> repo root
$Root = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $Root "app\main.py"))) {
    Write-Error "Could not find repo root (expected app\main.py under $Root)."
}
$SyncnetDir = if ($env:SYNCNET_DIR) { $env:SYNCNET_DIR } else { Join-Path $Root "syncnet_python" }

if ((Test-Path $SyncnetDir) -and -not (Test-Path (Join-Path $SyncnetDir "run_pipeline.py"))) {
    Write-Error "$SyncnetDir exists but is not a valid syncnet_python checkout. Remove the folder or set SYNCNET_DIR."
}

if (-not (Test-Path $SyncnetDir)) {
    Write-Host "Cloning SyncNet -> $SyncnetDir"
    git clone --depth 1 $RepoUrl $SyncnetDir
}

& (Join-Path $PSScriptRoot "download_model.ps1") -SyncnetRoot $SyncnetDir

if ($env:SKIP_SYNCNET_PATCH -eq "1") {
    Write-Host "SKIP_SYNCNET_PATCH=1 — skipping patch_syncnet_cpu.py"
} else {
    Write-Host "Applying CPU / NumPy compatibility patches..."
    python (Join-Path $Root "scripts\patch_syncnet_cpu.py") $SyncnetDir
}

Write-Host "SyncNet ready at $SyncnetDir"
