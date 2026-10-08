"""Konfigurasi global VidNote: path, batas, nama model."""
from __future__ import annotations

import os
from pathlib import Path

# Privasi: matikan telemetry Hugging Face. Warning symlink Windows tidak relevan.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Model default tidak butuh token; sembunyikan saran "set HF_TOKEN".
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Mode offline: tidak ada unduhan model sama sekali (VIDNOTE_OFFLINE=1 atau --offline).
OFFLINE = os.environ.get("VIDNOTE_OFFLINE") == "1"


def set_offline(on: bool = True) -> None:
    """Aktifkan mode offline. Panggil sebelum faster_whisper/huggingface_hub di-import."""
    global OFFLINE
    OFFLINE = on
    if on:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["VIDNOTE_OFFLINE"] = "1"


if OFFLINE:
    set_offline(True)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
WORK_DIR = ROOT_DIR / "work"
OUTPUT_DIR = ROOT_DIR / "output"

# Batas input
MAX_DURATION_SEC = 30 * 60
MAX_UPLOAD_BYTES = 1 * 1024**3  # 1 GB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
YOUTUBE_MAX_HEIGHT = 720
SUPPORTED_LANGS = ("id", "en")

# Anti bom dekompresi / file aneh
MIN_DIMENSION = 64
MAX_DIMENSION = 4096          # sisi terpanjang, setelah rotasi
MIN_DURATION_SEC = 1.0
# Container yang diterima (nilai format_name dari ffprobe, dipisah koma)
ALLOWED_CONTAINERS = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm"}

# Folder kerja: folder job lebih tua dari ini disapu saat start.
# Tidak disapu semua, supaya dua CLI yang jalan bersamaan (A/B test) tidak saling hapus.
STALE_JOB_HOURS = 6

# Timeout subprocess (detik)
FFPROBE_TIMEOUT = 30

# Audio untuk ASR/diarization
AUDIO_SAMPLE_RATE = 16000
AUDIO_FILENAME = "audio.wav"
# Target loudnorm (EBU R128). -16 LUFS umum untuk ucapan.
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
# Timeout ekstraksi = dasar + faktor x durasi video
AUDIO_TIMEOUT_BASE = 60
AUDIO_TIMEOUT_FACTOR = 1.0
# Di bawah level ini (dBFS RMS) audio dianggap hampir hening
SILENCE_RMS_DBFS = -60.0

# ASR (faster-whisper)
WHISPER_DIR = MODELS_DIR / "whisper"
ASR_BEAM_SIZE = 5
ASR_VAD_MIN_SILENCE_MS = 500
ASR_HALLUCINATION_SILENCE_SEC = 2.0
# Deteksi bahasa: jumlah potongan dan panjang tiap potongan (detik ucapan)
LANG_DETECT_CHUNKS = 3
LANG_DETECT_CHUNK_SEC = 30
# Jika total probabilitas id+en di bawah ini -> bahasa dianggap bukan id/en
LANG_MIN_SUPPORTED_PROB = 0.5
# Jika bahasa kedua punya porsi >= ini -> peringatan campur bahasa
LANG_MIXED_RATIO = 0.25
# Filter halusinasi
HALLU_NO_SPEECH_PROB = 0.6
HALLU_LOGPROB = -1.0
HALLU_COMPRESSION_RATIO = 2.4
# Diarization (sherpa-onnx, CPU). Model dari GitHub release resmi k2-fsa/sherpa-onnx.
# SHA256 dicatat saat unduhan pertama (4 Okt 2026); unduhan berikutnya wajib cocok.
DIAR_DIR = MODELS_DIR / "diarization"
_SHERPA_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
DIAR_SEGMENTATION = {
    "name": "pyannote-segmentation-3-0",
    "url": f"{_SHERPA_REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "sha256": "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    "member": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",  # file yang diambil dari arsip
    "member_sha256": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    "file": "pyannote-segmentation-3-0.onnx",  # lisensi MIT (CNRS)
}
DIAR_EMBEDDINGS = {
    # Pembanding A/B: CAM++ dilatih di VoxCeleb.
    "campplus-voxceleb": {
        "url": f"{_SHERPA_REL}/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "sha256": "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b",
        "file": "3dspeaker_campplus_sv_en_voxceleb_16k.onnx",
    },
    # Default: CAM++ zh+en (data "common", lebih besar).
    "campplus-zh-en": {
        "url": f"{_SHERPA_REL}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "sha256": "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
        "file": "3dspeaker_campplus_sv_zh_en_16k_common_advanced.onnx",
    },
}
# Default zh-en: di tes 4 Okt 2026 akurat di klip 4 pembicara (zh) dan dialog 2 suara (en),
# sedangkan voxceleb memecah 2 suara jadi 4. Belum dites di audio bahasa Indonesia.
DIAR_DEFAULT_EMBEDDING = "campplus-zh-en"

# Diarizer: "sherpa" (default, tanpa token) atau "pyannote" (opsional, butuh token HF sekali).
DIAR_BACKENDS = ("sherpa", "pyannote")
DIAR_DEFAULT_BACKEND = "sherpa"
PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_DIR = MODELS_DIR / "pyannote-community-1"   # lokasi clone offline
# Token dibaca dari env HF_TOKEN (mis. dari .env). Jangan pernah di-commit.
HF_TOKEN_ENV = "HF_TOKEN"
# Threshold clustering: lebih besar -> lebih sedikit pembicara. Di-tuning saat tes manual.
DIAR_THRESHOLD = 0.5
DIAR_MIN_DURATION_ON = 0.3
DIAR_MIN_DURATION_OFF = 0.5
DIAR_MAX_SPEAKERS = 8   # di atas ini diulang dengan num_clusters = batas
# Pembicara "hantu" (jingle/tawa): total bicara < max(MIN_SEC, MIN_RATIO x total) digabung
# ke pembicara terdekat. Tidak berlaku jika --num-speakers dipaksa.
DIAR_MINOR_MIN_SEC = 3.0
DIAR_MINOR_MIN_RATIO = 0.02

# Segmentasi subtitle/transkrip
SEG_MAX_GAP_SEC = 0.5          # jeda lebih dari ini -> potong
SEG_LINE_CHARS = {"landscape": 42, "portrait": 26}
SEG_MAX_LINES = 2              # satu cue maks 2 baris
SEG_MAX_CUE_SEC = 7.0
SEG_WORD_MAX_DIST_SEC = 1.0    # kata tanpa overlap: pakai turn terdekat dalam jarak ini
PARA_MAX_GAP_SEC = 2.0         # cue berurutan pembicara sama digabung jadi paragraf

# Burn hardsub
BURN_OUTPUT = "output.mp4"
BURN_NVENC = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
BURN_X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
BURN_AUDIO_BITRATE = "160k"
BURN_TIMEOUT_BASE = 120
BURN_TIMEOUT_FACTOR = {"gpu": 2.0, "cpu": 6.0}   # x durasi video
MAX_SUBS_BYTES = 20 * 1024**2
DOWNLOAD_MAX_BYTES = 500 * 1024**2

# Rantai fallback saat GPU OOM: (model, device). Default turbo sudah ringan (VRAM ~1 GB),
# jadi jika tetap OOM langsung turun ke CPU.
ASR_OOM_CHAIN = [("large-v3", "gpu"), ("large-v3-turbo", "gpu"), ("medium", "cpu")]

# Versi Python yang didukung (inklusif)
PYTHON_MIN = (3, 10)
PYTHON_MAX = (3, 12)

# Kebutuhan minimum sistem
MIN_RAM_GB = 8          # di bawah ini ditolak
RECOMMENDED_RAM_GB = 16
MIN_FREE_DISK_GB = 10   # model + video kerja + hardsub

# Model per device
MODELS = {
    "gpu": {
        # turbo dipilih setelah A/B (5 Okt 2026): WER ~sama dengan large-v3,
        # ~2,8x lebih cepat, VRAM separuh. large-v3 masih bisa lewat --asr-model.
        "asr": "large-v3-turbo",
        "asr_fallback": "large-v3-turbo",
        "asr_compute_type": "int8_float16",
        "llm": "gemma4:e4b",
    },
    "cpu": {
        "asr": "medium",
        "asr_fallback": "small",
        "asr_compute_type": "int8",
        "llm": "gemma4:e2b",
    },
}

# Ollama hanya diakses di loopback
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 900            # detik per request
SUM_CHUNK_SEC = 5 * 60          # map: ringkas per ~5 menit
SUM_NUM_CTX = 16384
SUM_TEMPERATURE = 0.2
# Batas ATAS poin akhir, menyesuaikan durasi (~2 poin/menit), dijepit MIN..MAX.
# Ini maksimal, bukan target: model boleh membuat lebih sedikit jika isi video sedikit.
SUM_POINTS_PER_MIN = 2
SUM_MIN_POINTS = 6
SUM_MAX_POINTS = 15
SUM_MAP_MAX_POINTS = 5          # poin per potongan ~5 menit (map)


def max_points_for(duration_sec: float) -> int:
    n = round((duration_sec / 60) * SUM_POINTS_PER_MIN)
    return max(SUM_MIN_POINTS, min(SUM_MAX_POINTS, n))
