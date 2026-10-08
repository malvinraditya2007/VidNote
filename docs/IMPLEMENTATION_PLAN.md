# VidNote: Implementation Plan (Fase 0)

> Dokumen ini adalah rencana kerja Fase 0. Acuan desain lengkap kini ada di bagian "Desain & keputusan teknis" pada `README.md`.

## Tujuan

CLI lokal `vidnote.py` yang mengubah video (file atau link YouTube, maks 30 menit, bahasa id/en) menjadi video hardsub, SRT/VTT/ASS, transkrip per pembicara, dan ringkasan. Semua model lokal, tanpa token atau API key. Repo dipublikasikan di GitHub, jadi harus jalan di hardware dan OS berbeda dengan opsi **GPU** dan **CPU**.

## Keputusan

| Aspek | Keputusan |
| --- | --- |
| OS resmi | Windows + Linux (GPU/CPU), macOS (CPU saja) |
| Python | 3.10-3.12 (dev: 3.11, berdampingan dengan 3.9 sistem) |
| ASR | GPU: Whisper `large-v3-turbo` `int8_float16` (default sejak A/B 5 Okt; `large-v3` via `--asr-model`). CPU: `medium` int8 (cadangan `small`) |
| Diarization | sherpa-onnx (pyannote-segmentation-3.0 ONNX + 3D-Speaker), tanpa token |
| Ringkasan | Ollama: Gemma 4 E4B (GPU) / E2B (CPU). Pembanding Qwen3 4B / 1.7B |
| Hardsub | ffmpeg `h264_nvenc` (GPU), fallback `libx264 veryfast` |
| YouTube | maks 720p (prioritas H.264+AAC), URL dikanonikkan dari video ID, `noplaylist`, tolak live. Butuh **Deno ≥ 2.3** + `yt-dlp[default]` (yt-dlp-ejs) untuk challenge JS YouTube |
| Upload | resolusi asli dipertahankan |
| Penyimpanan model | Whisper + sherpa-onnx di `models/` (gitignored). Ollama di penyimpanan bawaan Ollama |
| Folder kerja | `work/{uuid}/` (dihapus otomatis), hasil di `output/{nama}/` |
| Library CUDA | via pip (`requirements-gpu.txt`), opsional, tidak perlu CUDA Toolkit |
| Lisensi kode | MIT |
| Alur kerja | Berhenti setelah tiap task untuk tes manual |

Laptop uji developer: Windows, RTX 4050 Laptop 6 GB, driver CUDA 12.7. Platform lain (Linux, macOS, GPU 4 GB) ditandai **belum teruji** di README sampai ada yang mencoba.

## Struktur

```
vidnote.py                 CLI tipis (subcommand per tahap + run)
backend/config.py          path, batas, nama model
backend/device.py          deteksi CUDA + VRAM, pilih model, setup DLL CUDA
backend/doctor.py          cek environment
backend/pipeline/          probe, fetch_url, receive_file, audio, asr, diarize,
                           segment, subtitle, burn, summarize
tests/                     pytest + video sintetis dari ffmpeg
models/ work/ output/      gitignored
backend/runner.py          orkestrasi pipeline lengkap (dipakai CLI `run`, nanti FastAPI)
requirements*.in           dependency langsung
requirements*.txt          lockfile universal dengan hash (uv)
docs/                      plan ini + manual_test_log.md
```

Hasil antar tahap disimpan sebagai JSON di folder job, jadi tiap subcommand bisa dites terpisah.

## Task Fase 0

| # | Task | Demo |
| --- | --- | --- |
| 1 | Setup repo, environment, `doctor` | `python vidnote.py doctor` semua OK, GPU terdeteksi |
| 2 | Input file + probe (validasi tipe asli, durasi, stream, orientasi, folder kerja) | `vidnote.py probe video.mp4` |
| 3 | Input YouTube (allowlist, no-playlist, tolak live, cek durasi, maks 720p) | `vidnote.py fetch <url>` |
| 4 | Ekstraksi audio 16 kHz mono + loudnorm | `vidnote.py audio video.mp4` |
| 5 | ASR: VAD, word timestamps, deteksi id/en 3 potongan, filter halusinasi, OOM fallback | `vidnote.py asr video.mp4 --device gpu\|cpu` |
| 6 | Diarization sherpa-onnx, hash model dicek, `--num-speakers`, `--threshold` | `vidnote.py diarize video.mp4` |
| 7 | Gabung kata ↔ pembicara + segmentasi (jeda 0,5 dtk, 42/26 karakter) | `transcript.json`, `transcript.txt` |
| 8 | SRT/VTT/ASS, warna per pembicara, style landscape/portrait | Buka di VLC |
| 9 | Burn hardsub (NVENC → libx264), MP4 H.264 + AAC | `output.mp4` |
| 10 | Ringkasan Ollama map-reduce, JSON schema, timestamp, `--llm` | `summary.md` |
| 11 | `vidnote.py run`, mode offline, lockfile universal + hash, README | Satu perintah, GPU lalu CPU |

## Keamanan (berlaku di semua task)

- Subprocess selalu argumen list, tanpa `shell=True`.
- ffprobe/ffmpeg input upload: `-protocol_whitelist file,pipe`.
- Nama file asli diabaikan; simpan sebagai `work/{uuid}/input.ext`.
- Model diunduh hanya dari sumber resmi, hash dicek. `HF_HUB_OFFLINE=1` setelah ada.
- Ollama hanya diakses di `127.0.0.1:11434`. Transkrip diperlakukan sebagai data tidak tepercaya di prompt.

## Ditunda (outline saja)

- Opsi diarization pyannote community-1 (butuh token HF sekali).
- Proxy preview 480p (untuk frontend).
- **Opsi cookie YouTube (5 Okt):** `--cookies-from-browser` / `--cookies-file` di semua perintah YouTube + API, pesan error ramah.
- **Opsi b — pyannote community-1 (5 Okt): SELESAI.** `--diarizer pyannote` (opsional, extra `requirements-pyannote.txt`), model di-clone ke `models/pyannote-community-1/` (offline, tanpa token setelah unduh). A/B: pyannote lebih akurat tapi 4-5x lebih lambat di CPU → default tetap sherpa. torchcodec dihindari dengan memuat waveform di memori (fix Windows).
- ~~Fase 1: FastAPI + middleware Host/Origin/token, upload handler.~~ **SELESAI (5 Okt)**
- ~~Fase 2-6: integrasi pipeline ke API, SSE progress, cancel.~~ **SELESAI** (digabung ke Fase 1: `backend/app.py`, `jobqueue.py`, `security.py`; 176 test; dites end-to-end lewat HTTP)
- ~~Fase 7: frontend 1 halaman.~~ **SELESAI (5 Okt)** — `frontend/` (index.html, style.css, app.js tanpa framework), disajikan same-origin oleh FastAPI, token disuntik ke HTML. Tab Link/Upload, GPU/CPU, progres SSE, player hardsub, ringkasan + transkrip klik-untuk-seek, warna pembicara sinkron subtitle, XSS-safe (textContent). 181 test, dites di browser.
- Fase 8: evaluasi model dengan dataset (opsional).
- Fase 9: hardening, end-to-end test, setup script.

## Status

| Task | Status |
| --- | --- |
| 1 | Selesai, menunggu tes manual (Python 3.11.9, FFmpeg 9.0.2, Ollama 0.35.1, ctranslate2 4.8.2 + cuDNN 9 via pip, GPU smoke test OK) |
| 2 | Selesai, menunggu tes manual (30 test lolos; probe valid/tolak, rotasi, playlist palsu ditolak, folder kerja bersih) |
| 3 | Selesai, menunggu tes manual (68 test lolos; download nyata 4K → 1280x720 H.264; butuh Deno + yt-dlp-ejs) |
| 4 | Selesai, menunggu tes manual (74 test lolos; video 10,5 mnt → WAV 16 kHz mono dalam 7,7 dtk) |
| 5 | Selesai, menunggu tes manual bahasa Indonesia (95 test lolos; klip EN 19 dtk: GPU large-v3 13,3 dtk / +~2 GB VRAM, CPU medium 14,7 dtk, warm). OOM fallback hanya teruji lewat unit test |
| 6 | Selesai, menunggu tes manual bahasa Indonesia (106 test lolos; embedding default diganti ke `campplus-zh-en`: klip 4 pembicara zh akurasi 1,00, dialog TTS 2 suara en tepat 2; `campplus-voxceleb` memecah 2 suara jadi 4). Diarization jalan di CPU untuk kedua mode |
| 7 | Selesai, menunggu tes manual (118 test lolos; video Indonesia 2 orang 4:33 → 2 pembicara benar, total 94 dtk di GPU; pembicara "hantu" < 3 dtk/2% digabung otomatis) |
| 8 | Selesai, menunggu tes manual (129 test lolos; SRT/VTT/ASS dari transcript.json, diparse ffmpeg, render libass landscape/portrait dicek visual) |
| 9 | Selesai, menunggu tes manual (139 test lolos; video 4:32 di-burn libx264 dalam ~10 dtk, frame dicek visual). NVENC belum jalan di laptop dev: FFmpeg 9.0.2 butuh driver NVIDIA ≥ 610, terpasang 566.07 → fallback otomatis ke libx264, `doctor` memberi peringatan |
| 10 | Selesai, menunggu tes manual (150 test lolos; tag `gemma4:e4b` 6,6 GB / `gemma4:e2b` 4,6 GB terverifikasi. Video 4:33: E4B GPU 27,6 dtk, E2B CPU 32,3 dtk; map-reduce 3 bagian 36,5 dtk. E2B kadang salah timestamp) |
| 11 | Selesai (155 test lolos; `run` end-to-end video 4:32: GPU 2,1 mnt, CPU `--offline` 3,6 mnt; lockfile universal dengan hash via uv 0.12.23, terpasang ulang dengan `--require-hashes`; README + `docs/manual_test_log.md`) |

**Fase 0 selesai.** Berikutnya: tes manual dengan video sendiri (isi `docs/manual_test_log.md`), lalu Fase 1 (FastAPI) memakai `backend/runner.py`.
