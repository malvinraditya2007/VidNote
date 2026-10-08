# VidNote setup satu langkah (Windows / PowerShell).
# Pemakaian:
#   .\setup.ps1            # mode CPU
#   .\setup.ps1 -Gpu       # mode GPU (NVIDIA, CUDA 12.x)
#   .\setup.ps1 -Gpu -DownloadModels
param(
    [switch]$Gpu,
    [switch]$DownloadModels,
    [switch]$SkipTools
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Have($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

Write-Host "== VidNote setup (Windows) ==" -ForegroundColor Cyan

# 1. Program sistem lewat winget (opsional, bisa dilewati dengan -SkipTools).
if (-not $SkipTools) {
    if (-not (Have winget)) {
        Write-Warning "winget tidak ada. Instal manual: Python 3.11, FFmpeg, Ollama, Deno. Lihat README."
    } else {
        $pkgs = @("Python.Python.3.11", "Gyan.FFmpeg", "Ollama.Ollama", "DenoLand.Deno")
        foreach ($id in $pkgs) {
            Write-Host "-- winget install $id"
            winget install -e --id $id --accept-source-agreements --accept-package-agreements --disable-interactivity 2>$null
        }
        Write-Host "Catatan: buka terminal BARU setelah ini agar PATH ter-refresh." -ForegroundColor Yellow
    }
}

# 2. Virtual env Python 3.11.
if (-not (Test-Path ".venv")) {
    $py = if (Have py) { "py -3.11" } else { "python" }
    Write-Host "-- buat .venv ($py)"
    Invoke-Expression "$py -m venv .venv"
}
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet

# 3. Dependency (lockfile dengan hash).
$req = if ($Gpu) { "requirements-gpu.txt" } else { "requirements.txt" }
Write-Host "-- pip install -r $req"
& $venvPy -m pip install --require-hashes -r $req

# 4. Cek environment.
Write-Host "-- doctor"
& $venvPy vidnote.py doctor

# 5. Unduh model (opsional).
if ($DownloadModels) {
    $dev = if ($Gpu) { "gpu" } else { "cpu" }
    Write-Host "-- download-models --device $dev (bisa belasan GB)"
    & $venvPy vidnote.py download-models --device $dev
}

Write-Host ""
Write-Host "Selesai. Jalankan:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python vidnote.py serve        # buka http://127.0.0.1:8765"
Write-Host "  python vidnote.py run video.mp4 --device $(if ($Gpu) {'gpu'} else {'cpu'})"
