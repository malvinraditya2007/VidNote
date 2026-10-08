# VidNote

> **Video in, notes out.**

VidNote mengubah video (file atau link YouTube, maks 30 menit, bahasa Indonesia/Inggris) menjadi:

- video dengan **hardsub** berwarna per pembicara,
- subtitle **SRT / VTT / ASS**,
- **transkrip** per pembicara (Pembicara 1, 2, ...), terpecah rapi per pembicara **dan** per poin penting, bisa diklik untuk melompat ke posisi video,
- **ringkasan** poin penting + kesimpulan, dengan timestamp.

Semua diproses **lokal** di komputermu. Tanpa akun, tanpa API key, tanpa token. Setelah model terunduh, VidNote bisa jalan offline.

Dua cara pakai: **antarmuka web** (`serve`, jalur utama) atau **CLI** (`run`). Antarmuka web punya alur tiga langkah — Sumber → Proses → Hasil — dengan progres langsung, dan menampilkan video (landscape atau portrait), transkrip, serta ringkasan dalam satu halaman.

## Kebutuhan

| | Mode GPU | Mode CPU |
| --- | --- | --- |
| Hardware | GPU NVIDIA, driver dengan CUDA 12.x (cek `nvidia-smi`), VRAM ≥ 4 GB (disarankan 6 GB) | Semua PC/laptop, RAM ≥ 16 GB disarankan (min 8 GB) |
| ASR | Whisper `large-v3-turbo` (int8_float16) | Whisper `medium` (int8) |
| Ringkasan | Gemma 4 E4B (Ollama) | Gemma 4 E2B (Ollama) |
| Hardsub | NVENC, otomatis fallback ke libx264 | libx264 |

Disk: ±20 GB untuk semua model (Whisper ±4,5 GB, Gemma ±11 GB), plus ruang kerja ±3x ukuran video.

Software: **Python 3.10-3.12**, **FFmpeg**, **Ollama**, dan **Deno ≥ 2.3** (hanya untuk link YouTube).

### Status platform

| Platform | Status |
| --- | --- |
| Windows 11 + NVIDIA (RTX 4050 Laptop 6 GB) | Teruji, GPU dan CPU |
| Linux + NVIDIA | Belum teruji |
| macOS | Hanya mode CPU, belum teruji |
| GPU AMD / Intel | Tidak didukung untuk mode GPU (pakai `--device cpu`) |
| GPU dengan VRAM 4 GB | Belum teruji; ada fallback otomatis `large-v3` → `large-v3-turbo` → CPU |

## Mulai cepat

### Langkah 1 — Pasang (SEKALI SAJA, di komputer baru)

Setelah program sistem terpasang (lihat "Instalasi manual" di bawah), satu perintah menyiapkan sisanya:

```powershell
# Windows
.\setup.ps1 -Gpu -DownloadModels      # atau tanpa -Gpu untuk mode CPU
```

```bash
# Linux / macOS
./setup.sh --gpu --download-models     # atau tanpa --gpu
```

Script membuat `.venv`, memasang dependency (lockfile dengan hash), menjalankan `doctor`, dan (opsional) mengunduh model. **Langkah ini cukup dilakukan sekali.**

### Langkah 2 — Jalankan (SETIAP KALI mau pakai)

Tidak perlu setup lagi. Dari terminal di folder project:

```powershell
# Windows
.\.venv\Scripts\python.exe vidnote.py serve --open
```

```bash
# Linux / macOS
.venv/bin/python vidnote.py serve --open
```

`--open` membuka browser otomatis ke halaman VidNote. Tinggal tempel link / upload file dan klik **Proses**. Biarkan terminal tetap terbuka selama memakai; tekan `Ctrl+C` untuk mematikan.

## Instalasi manual

### 1. Program sistem

**Windows** (PowerShell):

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Gyan.FFmpeg
winget install -e --id Ollama.Ollama
winget install -e --id DenoLand.Deno
```

Lalu buka terminal baru supaya PATH terbaca.

**Linux (Debian/Ubuntu):**

```bash
sudo apt install python3.11 python3.11-venv ffmpeg
curl -fsSL https://ollama.com/install.sh | sh
curl -fsSL https://deno.land/install.sh | sh
```

**macOS:**

```bash
brew install python@3.11 ffmpeg ollama deno
```

### 2. Project

```powershell
git clone https://github.com/malvinraditya2007/VidNote.git VidNote
cd VidNote
py -3.11 -m venv .venv            # Linux/macOS: python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # Linux/macOS: source .venv/bin/activate

# Mode GPU (NVIDIA, Windows/Linux): termasuk cuBLAS + cuDNN 9 via pip, tidak perlu CUDA Toolkit
pip install -r requirements-gpu.txt
# Mode CPU saja (atau macOS):
pip install -r requirements.txt
```

Lockfile memakai hash (`--require-hashes` otomatis aktif), jadi paket yang isinya berbeda dari yang dipin akan ditolak pip.

**Linux + GPU:** library CUDA dari pip harus ada di `LD_LIBRARY_PATH` sebelum Python start. `python vidnote.py doctor` menampilkan perintah `export` yang tepat.

### 3. Cek dan unduh model

```powershell
python vidnote.py doctor                       # semua harus OK / WARN, tidak FAIL
python vidnote.py download-models --device gpu # atau cpu / all
```

## Pemakaian

### Antarmuka web (jalur utama)

```powershell
python vidnote.py serve --open      # buka browser otomatis ke http://127.0.0.1:8765
```

Alurnya tiga langkah:

1. **Sumber** — tempel link YouTube atau unggah file, pilih perangkat (GPU/CPU), bahasa, dan jumlah pembicara.
2. **Proses** — progres tujuh tahap langsung (cek video → ekstrak audio → transkripsi → pemisahan pembicara → transkrip & subtitle → ringkasan → hardsub), bisa dibatalkan.
3. **Hasil** — pemutar video (landscape/portrait), tombol unduh video hardsub & subtitle, transkrip per pembicara yang bisa diklik untuk melompat, serta kesimpulan dan poin penting.

Server hanya mengikat `127.0.0.1` dan menyuntik token sesi ke halaman; buka selalu lewat URL `serve`, bukan membuka file HTML langsung. Satu video diproses dalam satu waktu.

### CLI

```powershell
# Pipeline lengkap
python vidnote.py run "D:\video\rapat.mp4" --device gpu
python vidnote.py run "https://youtu.be/VIDEO_ID" --device cpu --lang id

# Opsi berguna
--lang auto|id|en       # default auto (deteksi dari 3 potongan, dibatasi id/en)
--num-speakers N        # paksa jumlah pembicara jika deteksi otomatis keliru
--no-summary / --no-burn
--yes                   # setuju otomatis turun model saat VRAM habis

# Tanpa internet sama sekali (model harus sudah diunduh, link YouTube ditolak)
python vidnote.py --offline run "D:\video\rapat.mp4" --device gpu

# Jika YouTube minta verifikasi "not a bot": pakai cookie login dari browser
# (tutup browser-nya dulu agar database cookie bisa dibaca)
python vidnote.py run "https://youtu.be/VIDEO_ID" --device gpu --cookies-from-browser chrome
python vidnote.py run "https://youtu.be/VIDEO_ID" --device gpu --cookies-file cookies.txt
```

Browser yang didukung untuk `--cookies-from-browser`: chrome, chromium, edge, firefox, brave, opera, vivaldi, safari.

Hasil ada di `output/<judul>-<waktu>/`:

| File | Isi |
| --- | --- |
| `video_hardsub.mp4` | Video H.264 + AAC dengan subtitle tertanam |
| `subs.srt`, `subs.vtt`, `subs.ass` | Subtitle (ASS = warna per pembicara) |
| `transcript.txt`, `transcript.json` | Transkrip per pembicara |
| `summary.md`, `summary.json` | Ringkasan + kesimpulan |
| `run.json` | Model, waktu per tahap, peringatan |
| `asr.json`, `diarization.json` | Data mentah untuk debug |

Waktu proses video 4,5 menit di RTX 4050 Laptop: **GPU 2,1 menit**, **CPU 3,6 menit**. Video lebih panjang dan CPU lebih lemah bisa jauh lebih lama.

Tiap tahap juga bisa dijalankan terpisah untuk tes/A-B: `probe`, `fetch`, `audio`, `asr`, `diarize`, `segment`, `subs`, `burn`, `summarize`. Lihat `python vidnote.py <perintah> --help`.

## Privasi dan keamanan

- Video, audio, transkrip, dan ringkasan **tidak pernah keluar dari komputermu**. Koneksi internet hanya dipakai untuk: unduh model (sekali), unduh video dari link YouTube, dan `ollama pull`.
- Telemetry Hugging Face dimatikan. Ollama hanya diakses di `127.0.0.1:11434`.
- Model diunduh dari sumber resmi: Hugging Face (Systran, mobiuslabsgmbh), GitHub release k2-fsa/sherpa-onnx (SHA256 dicek), registry Ollama.
- Input divalidasi (ekstensi, tipe asli via ffprobe, durasi, resolusi, ukuran ≤ 1 GB). Nama file asli diabaikan. Subprocess tanpa shell.
- Folder kerja `work/` dihapus otomatis, termasuk saat error atau Ctrl+C.
- `models/`, `work/`, `output/`, `.env` ada di `.gitignore`.

**Link YouTube:** mengunduh video dapat melanggar Ketentuan Layanan YouTube, dan isinya biasanya berhak cipta. Gunakan hanya untuk keperluan pribadi; jalur utama VidNote adalah file lokal.

## Keterbatasan

- Transkrip, pemisahan pembicara, dan ringkasan **dibuat AI dan bisa salah**. Tidak ada fitur edit.
- Nama orang dan istilah sering salah dengar (mis. "Vina" → "Pina").
- Pemisahan pembicara melemah saat orang bicara bersamaan, banyak pembicara, atau ada musik. Hasil clustering bisa berbeda antar run; pakai `--num-speakers` jika jumlahnya salah.
- Video campur Indonesia-Inggris akurasinya turun (ada peringatan).
- Ringkasan model kecil (E2B di CPU) kadang salah memberi timestamp.

## Troubleshooting

| Gejala | Solusi |
| --- | --- |
| `NVENC gagal ... Required: 13.1 Found: 12.2` | Driver NVIDIA terlalu lama untuk FFmpeg versi ini (butuh ≥ 610). VidNote otomatis pakai libx264; update driver jika ingin NVENC |
| `cudnn_ops64_9.dll` / `libcudnn_ops.so.9` tidak ditemukan | `pip install -r requirements-gpu.txt`; Linux: set `LD_LIBRARY_PATH` sesuai `doctor` |
| Link YouTube gagal / format hilang | Pastikan Deno ≥ 2.3 terpasang; update `yt-dlp` (lihat di bawah) |
| `Sign in to confirm you're not a bot` | YouTube membatasi IP-mu. Coba lagi nanti, atau pakai cookie login (lihat di bawah) |
| `Tutup browser itu dulu` saat pakai cookie | Database cookie terkunci. Tutup browser yang dipilih, lalu ulangi |
| `Ollama tidak merespons` | Jalankan aplikasi Ollama atau `ollama serve` |
| VRAM habis | Jawab `y` untuk turun ke `large-v3-turbo`/CPU, atau pakai `--device cpu` |

## Update dependency

Versi langsung ada di `requirements*.in`; lockfile `requirements*.txt` dibuat dengan [uv](https://github.com/astral-sh/uv):

```powershell
pip install -r requirements-dev.txt
foreach ($n in '','-gpu','-dev') { python -m uv pip compile "requirements$n.in" --universal --python-version 3.10 --generate-hashes --no-header -o "requirements$n.txt" }
python -m pytest -q
```

## Pengembangan ke depan

### Terjemahan lintas-bahasa (bahasa output ≠ bahasa video)

Rencana: pengguna memilih bahasa output di halaman Sumber, lalu transkrip-baca dan ringkasan ditampilkan dalam bahasa itu — misalnya video Inggris, catatan Indonesia. Belum diimplementasikan. Catatan desain supaya tidak terjebak:

**Yang bisa vs tidak:**

- **Ringkasan (Kesimpulan + Poin penting):** mudah — LLM lokal (Ollama) tinggal diminta menulis dalam bahasa target lewat `system_prompt` di `backend/pipeline/summarize.py`. Saat ini ringkasan selalu mengikuti bahasa video.
- **Transkrip:** butuh komponen terjemahan baru. Whisper hanya punya mode `translate` **ke Inggris**, jadi ID→EN bisa lewat Whisper, tapi EN→ID (dan arah lain) perlu mesin terjemahan terpisah — model MT khusus seperti **NLLB-200** atau **OPUS-MT (Marian)** memberi kualitas EN↔ID jauh lebih baik daripada LLM umum, dan keduanya bisa dijalankan via **CTranslate2** yang sudah dipakai VidNote (faster-whisper). Lisensi NLLB CC-BY-NC perlu diperhatikan untuk pemakaian non-komersial.

**Batasan yang menentukan arsitektur:**

- **Jangan terjemahkan hardsub.** Subtitle VidNote dipaku ke timestamp per-kata hasil Whisper; teks terjemahan punya jumlah/urutan kata berbeda, sehingga timing meleset dan baris berantakan. Memperbaikinya butuh re-timing (forced alignment) yang kompleks dan tetap tak sempurna. Hardsub di video **tetap bahasa asli**.
- **Posisikan terjemahan sebagai alat bantu, bukan sumber kebenaran.** Transkrip asli (akurat, termasuk slang) tetap tersedia; terjemahan tampil sebagai panel/subtitle terpisah dengan disclaimer "dibantu AI, bisa meleset". Slang, idiom, dan campur bahasa adalah titik lemah MT.
- **Perhatikan VRAM.** Di GPU 6-8 GB, model terjemahan akan berebut VRAM dengan Whisper dan LLM ringkasan; jalankan bergantian atau pilih model kecil (mis. NLLB distilled-600M).

**Arsitektur yang disarankan:** hardsub & transkrip asli = bahasa video (akurat); terjemahan = output tambahan untuk ringkasan + panel transkrip-baca + file SRT terpisah yang bisa dimatikan — bukan dibakar permanen ke video. Garap sebagai langkah terpisah yang bisa dites sendiri agar tak mengganggu pipeline yang sudah stabil.

## Desain & keputusan teknis

### Ruang lingkup

| Aspek | Keputusan |
| --- | --- |
| Input | Link YouTube **dan** upload file (`.mp4 .mov .mkv .webm`) |
| Durasi maks | 30 menit |
| Bahasa video | Indonesia dan Inggris |
| Orientasi | Landscape dan portrait |
| Pembicara | Banyak pembicara (Pembicara 1, 2, 3, ...), dipisah di transkrip |
| Perangkat | User memilih GPU atau CPU |
| Deploy | Lokal di PC masing-masing, tanpa server pusat |
| Frontend | Satu halaman HTML + CSS + JS, tanpa framework |
| Backend | Python (FastAPI) |
| Model | Open source, lokal, tanpa API key / akun / token (offline setelah unduhan awal) |
| Edit transkrip | Tidak ada (read-only) |

### Pipeline

```
[Link YT]  → validasi URL → cek durasi → download maks 720p ─┐
                                                             ├→ work/{job_id}/
[Upload]   → validasi ekstensi + ffprobe → simpan streaming ─┘
 → cek durasi ≤ 30 mnt, ada stream video & audio, resolusi wajar
 → deteksi landscape/portrait (ffprobe + rotation metadata)
 → ffmpeg: audio 16 kHz mono
 → faster-whisper: VAD (Silero), word timestamps, bahasa dibatasi id/en → filter halusinasi
 → [unload Whisper] → sherpa-onnx diarization (jumlah pembicara otomatis)
 → gabung kata ↔ pembicara, segmentasi per jeda & pergantian pembicara
 → SRT + VTT + ASS (warna per pembicara)
 → [unload diarization] → LLM lokal: ringkasan map-reduce (JSON + timestamp)
 → ffmpeg burn hardsub
```

Tahap berat dijalankan **berurutan** dan model di-unload antar tahap — penting untuk VRAM kecil. Saat GPU kehabisan memori, model turun bertahap: `large-v3` → `large-v3-turbo` → CPU.

### Keputusan model (4 Okt 2026)

Dasar: non-komersial, gratis, bisa diunduh, tanpa API key.

- **ASR: Whisper `large-v3` (bukan turbo) di GPU, `medium` di CPU.** Turbo lebih cepat dengan selisih WER kecil, tapi large-v3 unggul pada bahasa data-sedikit / aksen berat. Karena akurasi diprioritaskan dan tidak ada fitur edit, large-v3 jadi default; turbo adalah fallback saat VRAM kurang.
- **Diarization: sherpa-onnx (bukan pyannote).** pyannote community-1 mewajibkan login & token Hugging Face; versi ONNX pyannote-segmentation-3.0 + embedding 3D-Speaker tersedia tanpa token. Opsi upgrade pyannote community-1 (CC-BY-4.0, token sekali untuk unduh) tersedia untuk A/B test.
- **Ringkasan: Gemma 4 E4B (GPU) / E2B (CPU) via Ollama.** Ukuran kecil cocok untuk laptop; pembanding Qwen3 4B / 1.7B.

Catatan GPU: ctranslate2 terbaru butuh CUDA 12 + cuDNN 9 (error umum: `libcudnn_ops.so.9` tidak ditemukan jika salah versi).

### Keamanan

VidNote mengikat `127.0.0.1`, tapi `bind` saja tidak cukup — website lain di tab browser bisa menembak `localhost` (CSRF / DNS rebinding). Mitigasi:

- Cek header `Host` (hanya `127.0.0.1`/`localhost`) dan `Origin` pada request POST.
- Frontend disajikan FastAPI yang sama (same-origin), tanpa CORS.
- Token acak per sesi, disuntik ke halaman dan dikirim di header tiap request.
- Input: allowlist URL YouTube, verifikasi tipe file via ffprobe, batas ukuran ≤ 1 GB, nama file asli diabaikan, subprocess tanpa shell.
- Transkrip = data tidak tepercaya bagi LLM (anti prompt injection); output dirender `textContent` (anti XSS).
- Satu job satu waktu, folder kerja dihapus otomatis saat selesai/gagal/batal.

### Risiko utama

| Risiko | Mitigasi |
| --- | --- |
| VRAM kecil mepet | int8_float16, unload antar tahap, fallback turbo → CPU |
| Mode CPU lambat | Estimasi waktu di UI, tombol Batalkan |
| Whisper lemah pada slang / campur bahasa | Pilihan bahasa manual, peringatan + disclaimer |
| Diarization buruk saat pembicara banyak/overlap | Label generik, `--num-speakers`, disclaimer |
| yt-dlp break / YouTube memblokir | Update rutin; upload sebagai jalur utama |
| ToS YouTube & hak cipta | Upload jadi jalur utama, peringatan pada fitur link |

## Lisensi

Kode VidNote: **MIT** (lihat `LICENSE`). Untuk pemakaian pribadi, non-komersial.

Komponen pihak ketiga punya lisensinya sendiri:

| Komponen | Lisensi | Catatan |
| --- | --- | --- |
| Whisper (bobot model), faster-whisper, CTranslate2 | MIT | |
| Silero VAD | MIT | bawaan faster-whisper |
| pyannote segmentation-3.0 (versi ONNX) | MIT | dicek dari file LICENSE di arsip resmi |
| sherpa-onnx | Apache-2.0 | |
| 3D-Speaker CAM++ (embedding pembicara) | Apache-2.0 | lisensi repo 3D-Speaker; belum dicek per model |
| Gemma 4 | Apache-2.0 | |
| Ollama | MIT | |
| yt-dlp | Unlicense | |
| FFmpeg (build Gyan "full") | GPL | diinstal terpisah oleh user |
| Deno | MIT | |

Jika suatu hari dikomersialkan: audit ulang semua lisensi dan ToS YouTube.
