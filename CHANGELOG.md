# Changelog

## 0.1.0 — 2026-10-05

Rilis pertama. Pipeline lokal lengkap plus API dan frontend.

### Antarmuka
- Flag `serve --open`: membuka browser otomatis saat server siap.

### Diarizer alternatif (opsional)
- `--diarizer pyannote` (community-1) sebagai alternatif sherpa-onnx; extra `requirements-pyannote.txt` (PyTorch). Model di-clone ke `models/` untuk pemakaian offline tanpa token. Default tetap sherpa (lebih ringan); pyannote lebih akurat tapi lebih lambat di CPU.

### Opsi cookie YouTube
- `--cookies-from-browser <browser>` dan `--cookies-file <path>` di semua perintah yang mengambil YouTube, serta field `cookies_from_browser` di API. Mengatasi blokir "confirm you're not a bot". Pesan error ramah (mis. ingatkan tutup browser saat database cookie terkunci).

### Pipeline (CLI)
- `vidnote run <file|URL> --device gpu|cpu`: video → hardsub + SRT/VTT/ASS + transkrip per pembicara + ringkasan, dalam satu perintah.
- Subcommand per tahap untuk tes/A-B: `probe`, `fetch`, `audio`, `asr`, `diarize`, `segment`, `subs`, `burn`, `summarize`.
- `doctor` (cek environment), `download-models`, `serve`, dan `--offline`.

### Model (dipilih lewat A/B test)
- ASR: Whisper `large-v3-turbo` (GPU) / `medium` (CPU) via faster-whisper. VAD Silero, deteksi bahasa id/en, filter halusinasi, fallback OOM.
- Diarization: sherpa-onnx pyannote-segmentation-3.0 + 3D-Speaker CAM++ (zh-en), tanpa token. Penggabungan pembicara "hantu".
- Ringkasan: Gemma 4 E4B (GPU) / E2B (CPU) via Ollama. Map-reduce, batas poin dinamis (6–15), JSON tervalidasi, anti prompt injection.
- Hardsub: NVENC dengan fallback otomatis ke libx264.

### API & Frontend
- FastAPI lokal (`serve`), bind 127.0.0.1. Endpoint job, SSE progres, cancel, hasil, file (video dukung Range).
- Keamanan: cek Host/Origin, token sesi, CSP ketat, tanpa CORS. Anti-XSS di frontend.
- Frontend 1 halaman tanpa framework: tab Link/Upload, GPU/CPU, progres, player hardsub, transkrip klik-untuk-seek.

### Catatan
- Teruji di Windows + RTX 4050 Laptop (GPU & CPU). Linux/macOS/GPU 4 GB belum teruji.
- NVENC butuh driver NVIDIA ≥ 610 untuk FFmpeg 9.x; jika lebih lama, otomatis pakai libx264.
