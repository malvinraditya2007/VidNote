# Log Tes Manual VidNote (Fase 0)

Isi satu baris per run. Nilai 1-5 (5 = sangat bagus). Simpan `run.json` dari folder output jika ada masalah.

## Perangkat

| Item | Nilai |
| --- | --- |
| OS | |
| GPU / VRAM / driver | |
| CPU / RAM | |
| Versi VidNote (commit) | |

## Run end-to-end

| Tanggal | Video (judul/durasi) | Bahasa | Pembicara asli | Orientasi | Device | Total waktu | Transkrip (1-5) | Pembicara terdeteksi | Pembicara benar? | Ringkasan (1-5) | Fakta karangan? | Catatan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-05 | Ruang HRD Jobstreet, 4:32 | id | 2 | landscape | gpu | 2,1 mnt | 4 (nama salah dengar) | 2 | ya | 4 | tidak | NVENC fallback libx264 |
| 2026-10-05 | sama, file lokal, `--offline` | id | 2 | landscape | cpu | 3,6 mnt | | 2 | ya | | | E2B timestamp kadang meleset |
| | | | | | | | | | | | | |

## A/B ASR: large-v3 vs large-v3-turbo (GPU)

`python vidnote.py asr <video> --device gpu --out output\ab_v3` lalu `--asr-model large-v3-turbo --out output\ab_turbo`

Dites otomatis 2026-10-05 oleh Kiro. Referensi = subtitle Indonesia buatan kreator (bukan auto-generate). WER "mentah" menghitung beda ejaan (mie/mi, kalo/kalau) sebagai salah; WER "dinormalisasi" menyamakan varian ejaan umum. Subtitle kreator bisa sedikit diedit, jadi angka absolutnya kasar; perbandingan antar model tetap adil.

| Video | Model | Waktu ASR | VRAM | WER mentah | WER dinormalisasi |
| --- | --- | --- | --- | --- | --- |
| Mi Instan (mfjRsAbs6Ms, 4:10, vlog santai) | large-v3 (GPU) | 28 dtk | +2,1 GB | 13,8% | 7,2% |
| | large-v3-turbo (GPU) | 10 dtk | +1,0 GB | 14,5% | 7,7% |
| | medium (CPU) | 94 dtk | - | 10,3% | 4,9% |
| Kesalahan Belajar (1T2gaG5vPk8, 12:29, narasi) | large-v3 (GPU) | 72 dtk | +2,1 GB | 4,1% | 3,6% |
| | large-v3-turbo (GPU) | 26 dtk | +1,0 GB | 4,6% | 3,6% |
| | medium (CPU) | 285 dtk | - | 6,5% | 5,9% |

Kesimpulan: turbo ~2,8x lebih cepat dan setengah VRAM dengan akurasi hampir sama. Kesalahan yang tersisa kebanyakan salah dengar kata pendek ("suapan" → "sopan", "instan" → "insan"). Hanya 2 video, keduanya audio bersih; belum dites di audio bising atau banyak slang.

## A/B diarization: embedding

`python vidnote.py diarize <video> --embedding campplus-zh-en` vs `--embedding campplus-voxceleb`

| Video | Pembicara asli | zh-en | voxceleb | Pergantian rapi? | Catatan |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## A/B diarization: sherpa vs pyannote community-1

Dites 2026-10-05 oleh Kiro (CPU, karena PyTorch versi CPU terpasang). `--diarizer sherpa` vs `--diarizer pyannote`. Keduanya memakai audio 16 kHz yang sama.

| Video | Pembicara asli | sherpa | pyannote | Waktu sherpa | Waktu pyannote |
| --- | --- | --- | --- | --- | --- |
| Ruang HRD Jobstreet (4:32, 2 orang, studio) | 2 | 2 (via deteksi 10 → batasi 8 → gabung) | **2 (langsung bersih)** | 41 dtk | 162 dtk |
| Kesalahan Belajar (12:29, 1 narator + musik) | ~1 | **3 (terlalu pecah)** | **1 (benar)** | 87 dtk | 485 dtk |

Kesimpulan: **pyannote community-1 lebih akurat dan stabil** (deteksi jumlah pembicara lebih tepat, tanpa butuh batas/penggabungan paksa), tapi **4–5x lebih lambat di CPU**. Belum dites pyannote di GPU (PyTorch CPU; versi CUDA tidak cocok dengan lockfile universal).

Rekomendasi: default tetap **sherpa** (ringan, tanpa token, cukup baik). pyannote disediakan sebagai opsi `--diarizer pyannote` untuk video sulit atau saat akurasi lebih penting dari kecepatan. Keputusan ini sejajar dengan plan v4 (sherpa default, pyannote opsi upgrade).

## A/B ringkasan: Gemma 4 vs Qwen3

`ollama pull qwen3:4b` / `qwen3:1.7b`, lalu `python vidnote.py summarize <folder> --device gpu --llm qwen3:4b`

Dites 2026-10-05 oleh Kiro pada "Kesalahan Belajar" (12:29, map-reduce 3 bagian), dinilai dengan membaca transkrip.

| Model | Waktu | Bahasa natural (1-5) | Fakta karangan? | Timestamp tepat? | Cakupan | Catatan |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4:e4b (GPU) | 42 dtk | 5 | tidak | ya | awal-akhir, bagian sejarah (07:54-10:33) terlewat | parafrase rapi, kesimpulan bagus |
| qwen3:4b (GPU) | 70 dtk | 2 | tidak | ya | hanya 00:00-05:31 | sebagian besar poin menyalin kalimat transkrip, ada poin dobel |
| gemma4:e2b (CPU) | 72 dtk | 4 | tidak | ya | awal-akhir, bagian sejarah terlewat | poin terakhir menyalin kalimat panjang |
| qwen3:1.7b (CPU) | 77 dtk | 3 | tidak | ya | sebagian | poin generik, kesimpulan diawali "Poin-poin ini..." |

Kesimpulan: Gemma 4 tetap default (lebih cepat, lebih ringkas, cakupan lebih luas).

Tindak lanjut (5 Okt): batas poin diubah jadi dinamis (~2/menit, 6-15). Uji ulang E4B pada video 12 menit → 15 poin, bagian sejarah [08:12] yang sebelumnya hilang kini masuk, cakupan awal-akhir lengkap, tanpa fakta karangan. Batas 8 poin memang terlalu ketat untuk video panjang.

Catatan diarization video ini: narasi dengan musik latar terdeteksi 3 pembicara, dan pergantiannya beberapa kali jatuh di tengah kalimat (mis. 03:23 → 03:30). Jumlah narator asli belum diverifikasi.

## Kasus khusus

| Kasus | Hasil | Catatan |
| --- | --- | --- |
| Video portrait (HP, rotasi metadata) | | |
| Video campur id/en | | |
| Audio bising / musik latar | | |
| 3+ pembicara, saling potong | | |
| Tepat 30 menit / > 30 menit | | |
| Ctrl+C di tengah tiap tahap (folder `work/` bersih?) | | |
