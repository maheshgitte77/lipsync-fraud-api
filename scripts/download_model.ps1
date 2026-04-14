# Same assets as syncnet_python/download_model.sh — run on Windows (PowerShell).
# Usage (from repo root or anywhere):
#   .\scripts\download_model.ps1
#   .\scripts\download_model.ps1 -SyncnetRoot "C:\path\to\syncnet_python"

param(
    [string]$SyncnetRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $SyncnetRoot) {
    $SyncnetRoot = Join-Path (Split-Path $PSScriptRoot -Parent) "syncnet_python"
}

if (-not (Test-Path $SyncnetRoot -PathType Container)) {
    Write-Error "SyncNet folder not found: $SyncnetRoot"
}

$SyncnetRoot = (Resolve-Path $SyncnetRoot).Path
$dataDir = Join-Path $SyncnetRoot "data"
$weightsDir = Join-Path $SyncnetRoot "detectors\s3fd\weights"

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $weightsDir | Out-Null

$files = @(
    @{
        Url  = "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
        Dest = Join-Path $dataDir "syncnet_v2.model"
    },
    @{
        Url  = "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi"
        Dest = Join-Path $dataDir "example.avi"
    },
    @{
        Url  = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth"
        Dest = Join-Path $weightsDir "sfd_face.pth"
    }
)

foreach ($item in $files) {
    Write-Host "Downloading $($item.Url) -> $($item.Dest)"
    Invoke-WebRequest -Uri $item.Url -OutFile $item.Dest -UseBasicParsing
}

Write-Host "Done. Weights are under $dataDir and $weightsDir"
