"""
Central configuration for the meeting-notes pipeline.
All tuneable knobs live here so nothing is scattered across modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # picks up .env for HF_TOKEN, etc.

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.resolve()
UPLOAD_DIR     = BASE_DIR / "uploads"      # raw uploaded / downloaded video files
WORK_DIR       = BASE_DIR / "workdir"      # intermediate per-job artefacts
OUTPUT_DIR     = BASE_DIR / "outputs"      # final PDFs
DB_PATH        = BASE_DIR / "storage" / "meetings.db"
STATIC_DIR     = BASE_DIR / "api" / "static"

for _d in (UPLOAD_DIR, WORK_DIR, OUTPUT_DIR, DB_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)


# ── Accepted input formats ────────────────────────────────────────────────────

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm"}


# ── Audio / Transcription ─────────────────────────────────────────────────────

WHISPER_MODEL       = "large-v3"          # options: tiny, base, small, medium, large-v3
WHISPER_LANGUAGE    = None                # None → auto-detect; "tr", "en", …
WHISPER_DEVICE      = "cuda"              # "cpu" fallback handled at runtime
WHISPER_BATCH_SIZE  = 16                  # reduce if VRAM is tight

# pyannote speaker diarisation
PYANNOTE_MODEL      = "pyannote/speaker-diarization-3.1"
HF_TOKEN            = os.getenv("HF_TOKEN", "")   # required for pyannote gated model
MIN_SPEAKERS        = 1
MAX_SPEAKERS        = 10


# ── Frame Extraction ──────────────────────────────────────────────────────────

FRAME_SAMPLE_INTERVAL  = 0.5   # seconds between sampled frames
CLIP_SIMILARITY_THRESH = 0.95  # cosine similarity above this → skip frame (same screen)
CLIP_CONTENT_THRESH    = 0.20  # min similarity to "content" prompts to keep frame


# ── CLIP (frame deduplication) ────────────────────────────────────────────────

CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"
CLIP_DEVICE       = "cuda"


# ── VLM — Qwen2.5-VL (screen understanding) ──────────────────────────────────

VLM_MODEL_NAME       = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_DEVICE_MAP       = "auto"             # spreads across available GPUs/CPU
VLM_LOAD_IN_4BIT     = True              # bitsandbytes 4-bit quantisation
VLM_MAX_NEW_TOKENS   = 512
VLM_PROMPT           = (
    "Bu ekran görüntüsünde ne var? Kısaca ve net açıkla.\n"
    "Eğer terminal veya komut satırı varsa komutları ve çıktıları aynen yaz.\n"
    "Eğer kod varsa kodu aynen yaz.\n"
    "Eğer tablo, grafik veya diyagram varsa içeriğini özetle.\n"
    "Eğer bir arayüz veya uygulama ekranı varsa ne gösterdiğini açıkla.\n"
    "Türkçe cevap ver."
)


# ── LLM — Mistral 7B (note generation) ───────────────────────────────────────

LLM_MODEL_NAME         = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_DEVICE_MAP         = "auto"
LLM_LOAD_IN_4BIT       = True
LLM_MAX_NEW_TOKENS     = 2048
LLM_TEMPERATURE        = 0.3
LLM_TOP_P              = 0.9
LLM_MAX_CHUNK_TOKENS   = 3000   # context window'a sığacak maksimum input token tahmini
                                 # Mistral 7B context: 32k; prompt overhead ~500 token bırakarak
                                 # 3000 güvenli bir alt sınır — büyütmek hızı düşürür


# ── Note generation prompt ────────────────────────────────────────────────────

NOTE_SYSTEM_PROMPT = """\
Sen bir toplantı asistanısın. Aşağıda kronolojik sıraya dizilmiş toplantı \
transkripti ve ekran açıklamaları verilmiştir. Bu bilgilerden yapılandırılmış \
bir toplantı notu üret.

Çıktı şu bölümleri içermeli:
1. Genel Özet (3-5 cümle)
2. Konuşmacı Bazlı Notlar (her konuşmacının katkısı maddeler halinde)
3. Alınan Kararlar
4. Action Item'lar (sorumlu kişi ve varsa tarih ile)
5. Teknik Detaylar (kod, komut veya ekran paylaşımından elde edilenler)

Yanıtını Türkçe ver.\
"""


# ── PDF ───────────────────────────────────────────────────────────────────────

PDF_CSS_FILE   = BASE_DIR / "output" / "style.css"    # optional custom stylesheet
PDF_FONT_URLS  = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
]


# ── YouTube / yt-dlp ──────────────────────────────────────────────────────────

YTDLP_FORMAT          = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
YTDLP_USE_AUTOSUB     = True    # use YouTube auto-captions when available
YTDLP_AUTOSUB_LANG    = "tr"    # preferred caption language; falls back to "en"


# ── FastAPI / Server ──────────────────────────────────────────────────────────

API_HOST          = os.getenv("API_HOST", "0.0.0.0")
API_PORT          = int(os.getenv("API_PORT", "8000"))
API_WORKERS       = 1           # keep at 1; models are not fork-safe
MAX_UPLOAD_MB     = 2048        # reject uploads larger than this
CORS_ORIGINS      = ["*"]       # tighten in production


# ── Pipeline concurrency ──────────────────────────────────────────────────────

# Audio and frame pipelines run sequentially by default.
# Set to True only if you have enough VRAM to keep all models loaded at once.
PARALLEL_AUDIO_VIDEO = False


# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")   # DEBUG | INFO | WARNING | ERROR
