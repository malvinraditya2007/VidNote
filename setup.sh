#!/usr/bin/env bash
# VidNote setup satu langkah (Linux / macOS).
# Pemakaian:
#   ./setup.sh                 # mode CPU
#   ./setup.sh --gpu           # mode GPU (NVIDIA, CUDA 12.x; Linux saja)
#   ./setup.sh --gpu --download-models
set -euo pipefail
cd "$(dirname "$0")"

GPU=0
DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    --download-models) DOWNLOAD=1 ;;
    *) echo "argumen tak dikenal: $arg"; exit 1 ;;
  esac
done

echo "== VidNote setup =="

# 1. Cek program sistem (instalasi diserahkan ke user; lihat README).
missing=""
for tool in python3 ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
command -v ollama >/dev/null 2>&1 || echo "Catatan: 'ollama' tidak ditemukan (perlu untuk ringkasan)."
command -v deno   >/dev/null 2>&1 || echo "Catatan: 'deno' tidak ditemukan (perlu untuk link YouTube)."
if [ -n "$missing" ]; then
  echo "WAJIB tapi tidak ada:$missing"
  echo "  Debian/Ubuntu: sudo apt install python3.11 python3.11-venv ffmpeg"
  echo "  macOS:         brew install python@3.11 ffmpeg"
  exit 1
fi

# 2. venv.
PY=python3
if command -v python3.11 >/dev/null 2>&1; then PY=python3.11; fi
[ -d .venv ] || { echo "-- buat .venv ($PY)"; "$PY" -m venv .venv; }
VENV=./.venv/bin/python
"$VENV" -m pip install --upgrade pip --quiet

# 3. Dependency.
REQ=requirements.txt
[ "$GPU" = "1" ] && REQ=requirements-gpu.txt
echo "-- pip install -r $REQ"
"$VENV" -m pip install --require-hashes -r "$REQ"

# 4. Doctor (menampilkan juga LD_LIBRARY_PATH untuk GPU Linux bila perlu).
echo "-- doctor"
"$VENV" vidnote.py doctor || true

# 5. Model (opsional).
if [ "$DOWNLOAD" = "1" ]; then
  DEV=cpu; [ "$GPU" = "1" ] && DEV=gpu
  echo "-- download-models --device $DEV (bisa belasan GB)"
  "$VENV" vidnote.py download-models --device "$DEV"
fi

echo ""
echo "Selesai. Jalankan:"
echo "  source .venv/bin/activate"
echo "  python vidnote.py serve        # buka http://127.0.0.1:8765"
