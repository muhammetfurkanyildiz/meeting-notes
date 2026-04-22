"""
input_handler.py
────────────────
İki kaynaktan girdi alır:
  1. Yerel video dosyası (.mp4 / .mkv / .avi / .webm)
  2. YouTube linki  →  yt-dlp ile video + isteğe bağlı otomatik altyazı

Her iki durumda da:
  • Standart bir VideoInput dataclass döner.
  • ffmpeg aracılığıyla ses track'i ayrı bir .wav olarak kaydedilir.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ffmpeg
import yt_dlp

import config

# ── Logger ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("input_handler")


# ── Veri modeli ───────────────────────────────────────────────────────────────

@dataclass
class VideoInput:
    """Pipeline boyunca taşınan standart girdi nesnesi."""

    video_path: str                          # mutlak yol, .mp4 / .mkv / ...
    audio_path: str                          # ayrıştırılmış .wav (mono, 16 kHz)
    source_type: Literal["file", "youtube"]
    title: str
    duration: float                          # saniye cinsinden
    existing_transcript: str | None = None  # YouTube altyazısı varsa ham metin
    metadata: dict = field(default_factory=dict)  # ek bilgiler (uploader, url, …)


# ── Hata sınıfları ────────────────────────────────────────────────────────────

class InputHandlerError(Exception):
    """Genel girdi işleme hatası."""


class UnsupportedFormatError(InputHandlerError):
    """Desteklenmeyen dosya uzantısı."""


class VideoTooLargeError(InputHandlerError):
    """Dosya boyutu MAX_UPLOAD_MB sınırını aşıyor."""


class DownloadError(InputHandlerError):
    """yt-dlp indirme hatası."""


class AudioExtractionError(InputHandlerError):
    """ffmpeg ses ayırma hatası."""


# ── Ana arayüz fonksiyonları ──────────────────────────────────────────────────

def handle_file(file_path: str | Path) -> VideoInput:
    """
    Yerel bir video dosyasını alır, doğrular ve VideoInput döner.

    Parameters
    ----------
    file_path:
        İşlenecek video dosyasının yolu.

    Raises
    ------
    UnsupportedFormatError  – uzantı ALLOWED_VIDEO_EXTENSIONS içinde değilse
    VideoTooLargeError      – dosya MAX_UPLOAD_MB'den büyükse
    AudioExtractionError    – ffmpeg ses ayıramazsa
    InputHandlerError       – dosya bulunamazsa veya okunamazsa
    """
    path = Path(file_path).resolve()

    logger.info("Dosya işleniyor: %s", path)

    # ── Temel kontroller ──────────────────────────────────────────────────────
    if not path.exists():
        raise InputHandlerError(f"Dosya bulunamadı: {path}")
    if not path.is_file():
        raise InputHandlerError(f"Belirtilen yol bir dosya değil: {path}")

    ext = path.suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Desteklenmeyen format '{ext}'. "
            f"Kabul edilen formatlar: {', '.join(sorted(config.ALLOWED_VIDEO_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 ** 2)
    if size_mb > config.MAX_UPLOAD_MB:
        raise VideoTooLargeError(
            f"Dosya boyutu {size_mb:.1f} MB, izin verilen üst sınır {config.MAX_UPLOAD_MB} MB."
        )

    logger.info("Dosya doğrulandı — boyut: %.1f MB, format: %s", size_mb, ext)

    # ── Video kopyasını uploads/ altına taşı ─────────────────────────────────
    dest_path = _unique_dest(path.name)
    if not dest_path.exists():
        shutil.copy2(path, dest_path)
        logger.info("Dosya uploads/ dizinine kopyalandı: %s", dest_path.name)
    else:
        logger.debug("Aynı isimde dosya zaten mevcut, kopyalanmadı.")

    # ── Süre ve başlık ───────────────────────────────────────────────────────
    duration = _probe_duration(dest_path)
    title = path.stem

    # ── Ses ayırma ────────────────────────────────────────────────────────────
    audio_path = _extract_audio(dest_path)

    logger.info(
        "Dosya hazır — başlık: '%s', süre: %s",
        title,
        _fmt_duration(duration),
    )

    return VideoInput(
        video_path=str(dest_path),
        audio_path=str(audio_path),
        source_type="file",
        title=title,
        duration=duration,
    )


def handle_youtube(url: str) -> VideoInput:
    """
    YouTube URL'sini indirir, varsa otomatik altyazıyı da alır ve VideoInput döner.

    Parameters
    ----------
    url:
        YouTube video URL'si.

    Raises
    ------
    DownloadError          – yt-dlp indirme başarısız olursa
    AudioExtractionError   – ffmpeg ses ayıramazsa
    InputHandlerError      – URL geçersizse
    """
    if not _is_youtube_url(url):
        raise InputHandlerError(
            f"Geçersiz YouTube URL'si: '{url}'. "
            "Lütfen 'https://www.youtube.com/watch?v=...' veya 'https://youtu.be/...' formatında bir URL girin."
        )

    logger.info("YouTube videosu indiriliyor: %s", url)

    job_id   = uuid.uuid4().hex[:8]
    out_dir  = config.UPLOAD_DIR / f"yt_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path, title, duration, metadata = _download_video(url, out_dir)
    transcript                            = _download_transcript(url, out_dir)

    if transcript:
        logger.info("Otomatik altyazı bulundu ve indirildi (%d karakter).", len(transcript))
    else:
        logger.info("Otomatik altyazı bulunamadı; Whisper ile transkript oluşturulacak.")

    audio_path = _extract_audio(video_path)

    logger.info(
        "YouTube videosu hazır — başlık: '%s', süre: %s",
        title,
        _fmt_duration(duration),
    )

    return VideoInput(
        video_path=str(video_path),
        audio_path=str(audio_path),
        source_type="youtube",
        title=title,
        duration=duration,
        existing_transcript=transcript,
        metadata=metadata,
    )


# ── İç yardımcı fonksiyonlar ──────────────────────────────────────────────────

def _unique_dest(filename: str) -> Path:
    """uploads/ altında çakışmayı önlemek için gerekirse UUID prefix ekler."""
    dest = config.UPLOAD_DIR / filename
    if dest.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        dest = config.UPLOAD_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
    return dest


def _probe_duration(video_path: Path) -> float:
    """ffprobe ile videonun süresini saniye olarak döner."""
    try:
        probe = ffmpeg.probe(str(video_path))
        return float(probe["format"]["duration"])
    except ffmpeg.Error as exc:
        raise InputHandlerError(
            f"Video süresi okunamadı ({video_path.name}): {exc.stderr.decode(errors='replace')}"
        ) from exc
    except (KeyError, ValueError) as exc:
        raise InputHandlerError(
            f"ffprobe çıktısı beklenmeyen formatta ({video_path.name}): {exc}"
        ) from exc


def _extract_audio(video_path: Path) -> Path:
    """
    ffmpeg ile video dosyasından mono 16 kHz WAV ses dosyası üretir.
    Çıktı workdir/ altına kaydedilir.
    """
    audio_path = config.WORK_DIR / (video_path.stem + "_audio.wav")

    if audio_path.exists():
        logger.debug("Ses dosyası zaten mevcut, yeniden oluşturulmadı: %s", audio_path.name)
        return audio_path

    logger.info("Ses track'i ayrıştırılıyor → %s", audio_path.name)

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(audio_path),
                ac=1,          # mono
                ar=16000,      # 16 kHz  (Whisper için ideal)
                acodec="pcm_s16le",
                vn=None,       # video stream'i dahil etme
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        # Geçici yarım dosyayı temizle
        audio_path.unlink(missing_ok=True)
        raise AudioExtractionError(
            f"Ses ayırma başarısız ({video_path.name}): {exc.stderr.decode(errors='replace')}"
        ) from exc

    size_mb = audio_path.stat().st_size / (1024 ** 2)
    logger.info("Ses dosyası oluşturuldu — boyut: %.1f MB", size_mb)
    return audio_path


def _download_video(url: str, out_dir: Path) -> tuple[Path, str, float, dict]:
    """
    yt-dlp ile videoyu indirir.

    Returns
    -------
    (video_path, title, duration_secs, metadata_dict)
    """
    # İndirmeden önce metadata al
    meta = _fetch_metadata(url)
    title    = meta.get("title", "youtube_video")
    duration = float(meta.get("duration") or 0.0)

    safe_title  = _safe_filename(title)
    output_tmpl = str(out_dir / f"{safe_title}.%(ext)s")

    ydl_opts: dict = {
        **_ydlp_base_opts(),
        "format":              config.YTDLP_FORMAT,
        "outtmpl":             output_tmpl,
        "merge_output_format": "mp4",
    }

    logger.info("Video indiriliyor — '%s'", title)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"Video indirilemedi: {exc}") from exc

    # İndirilen dosyayı bul (.mp4 garantili)
    candidates = list(out_dir.glob("*.mp4"))
    if not candidates:
        # Farklı uzantı kalmış olabilir
        candidates = [p for p in out_dir.iterdir() if p.is_file() and not p.suffix == ".json"]
    if not candidates:
        raise DownloadError(f"İndirme tamamlandı ancak çıktı dosyası bulunamadı: {out_dir}")

    video_path = max(candidates, key=lambda p: p.stat().st_size)
    logger.info("İndirme tamamlandı: %s (%.1f MB)", video_path.name, video_path.stat().st_size / 1024**2)

    # Süre ffprobe ile doğrula (metadata bazen eksik olabilir)
    if duration == 0.0:
        try:
            duration = _probe_duration(video_path)
        except InputHandlerError:
            pass

    return video_path, title, duration, meta


def _ydlp_base_opts() -> dict:
    """Tüm yt-dlp çağrıları için ortak seçenekler (çerez desteği dahil)."""
    opts: dict = {"quiet": True, "no_warnings": True}
    browser = getattr(config, "YTDLP_COOKIES_FROM_BROWSER", "")
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    return opts


def _fetch_metadata(url: str) -> dict:
    """yt-dlp ile indirme yapmadan video metadata'sını çeker."""
    ydl_opts = {
        **_ydlp_base_opts(),
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info or {}
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"Video bilgisi alınamadı: {exc}") from exc


def _download_transcript(url: str, out_dir: Path) -> str | None:
    """
    YouTube'un otomatik altyazısını indirir; önce config'deki dili dener,
    bulamazsa İngilizce'ye düşer.  Hiç altyazı yoksa None döner.

    Ham metin VTT → düz metin olarak döner (zaman damgaları çıkarılır).
    """
    if not config.YTDLP_USE_AUTOSUB:
        return None

    preferred_langs = [config.YTDLP_AUTOSUB_LANG, "en"]

    for lang in preferred_langs:
        sub_path = _try_download_sub(url, out_dir, lang)
        if sub_path:
            text = _vtt_to_text(sub_path)
            sub_path.unlink(missing_ok=True)  # ham dosyayı temizle
            if text.strip():
                logger.debug("Altyazı dili kullanıldı: %s", lang)
                return text

    return None


def _try_download_sub(url: str, out_dir: Path, lang: str) -> Path | None:
    """Belirtilen dilde otomatik altyazıyı indir; bulamazsa None döner."""
    sub_tmpl = str(out_dir / "sub.%(ext)s")
    ydl_opts = {
        **_ydlp_base_opts(),
        "skip_download":     True,
        "writeautomaticsub": True,
        "subtitleslangs":    [lang],
        "subtitlesformat":   "vtt",
        "outtmpl":           sub_tmpl,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError:
        return None

    candidates = list(out_dir.glob("*.vtt"))
    return candidates[0] if candidates else None


def _vtt_to_text(vtt_path: Path) -> str:
    """
    WebVTT dosyasından sadece konuşma metnini çıkarır.
    Zaman damgaları, NOTE blokları ve boş satırlar atılır.
    Yinelenen ardışık satırlar birleştirilir (YouTube VTT'de çift satır olur).
    """
    raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    prev = ""

    for line in raw.splitlines():
        line = line.strip()
        # Başlık, zaman damgası ve NOTE satırlarını atla
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("NOTE")
            or "-->" in line
            or re.match(r"^\d+$", line)          # sadece rakamdan oluşan satır numarası
        ):
            continue

        # HTML etiketlerini temizle (<c>, </c>, <00:00:00.000> gibi)
        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line:
            continue

        # YouTube VTT'deki yinelenen ardışık satırları kaldır
        if line != prev:
            lines.append(line)
            prev = line

    return " ".join(lines)


def _is_youtube_url(url: str) -> bool:
    """Basit regex ile YouTube URL'si mi diye kontrol eder."""
    pattern = re.compile(
        r"^(https?://)?"
        r"(www\.)?"
        r"(youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/shorts/)"
        r"[\w\-]+"
    )
    return bool(pattern.search(url))


def _safe_filename(name: str) -> str:
    """Dosya adı için geçersiz karakterleri temizler, 80 karaktere kısaltır."""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:80]


def _fmt_duration(seconds: float) -> str:
    """123.4 → '2d 3s' gibi okunabilir süre metni."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}s {m}d {s:02d}sn"
    if m:
        return f"{m}d {s:02d}sn"
    return f"{s}sn"
