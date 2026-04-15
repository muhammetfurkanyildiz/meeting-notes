"""
audio_processor.py
──────────────────
VideoInput alır, üç aşamada işler:

  1. Whisper Large-v3  →  List[TranscriptSegment]   (metin + timestamp)
  2. pyannote.audio    →  List[SpeakerSegment]       (konuşmacı + timestamp)
  3. Birleştirme       →  List[AnnotatedSegment]     (metin + konuşmacı + timestamp)

Modeller lazy-load edilir; ilk çağrıda GPU/CPU otomatik algılanır,
sonraki çağrılarda modüler önbellek kullanılır (process ömrü boyunca tekrar yüklenmez).

existing_transcript varsa (YouTube altyazısı) Whisper adımı atlanır,
altyazı metni timestamp'siz segment olarak wrap edilir.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

import config
from pipeline.input_handler import VideoInput

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("audio_processor")

# ── Cihaz tespiti ─────────────────────────────────────────────────────────────

def _resolve_device(preferred: str) -> str:
    """
    config'deki tercih edilen cihazı doğrular; CUDA yoksa CPU'ya düşer.
    """
    if preferred == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA bulunamadı; %s CPU modunda çalışacak. "
            "İşlem çok daha yavaş olacaktır.",
            preferred,
        )
        return "cpu"
    return preferred


# ── Veri modelleri ────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """Whisper çıktısının tek bir cümlesi."""
    start: float
    end: float
    text: str
    language: str


@dataclass
class SpeakerSegment:
    """pyannote diarisation çıktısının tek bir konuşmacı bloğu."""
    start: float
    end: float
    speaker: str          # "SPEAKER_00", "SPEAKER_01", …  →  etiket dönüşümü aşağıda


@dataclass
class AnnotatedSegment:
    """Birleştirilmiş nihai segment: metin + konuşmacı + zaman."""
    start: float
    end: float
    text: str
    speaker: str          # "SPEAKER_A", "SPEAKER_B", …
    language: str


# ── Hata sınıfları ────────────────────────────────────────────────────────────

class AudioProcessorError(Exception):
    """Genel ses işleme hatası."""


class WhisperError(AudioProcessorError):
    """Whisper transkripsiyon hatası."""


class DiarizationError(AudioProcessorError):
    """pyannote diarisation hatası."""


# ── Lazy-load model önbelleği ─────────────────────────────────────────────────
# Modüler değişkenler; process boyunca sadece bir kez yüklenir.

_whisper_model = None      # whisper.model.Whisper
_pyannote_pipeline = None  # pyannote.audio.Pipeline


def _get_whisper():
    """Whisper modelini döner; gerekiyorsa yükler."""
    global _whisper_model
    if _whisper_model is None:
        import whisper  # geç import — ağır bağımlılık

        device = _resolve_device(config.WHISPER_DEVICE)
        logger.info(
            "Whisper '%s' modeli yükleniyor (cihaz: %s) …",
            config.WHISPER_MODEL,
            device,
        )
        try:
            _whisper_model = whisper.load_model(config.WHISPER_MODEL, device=device)
        except Exception as exc:
            raise WhisperError(f"Whisper modeli yüklenemedi: {exc}") from exc
        logger.info("Whisper modeli yüklendi.")
    return _whisper_model


def _get_pyannote():
    """pyannote diarisation pipeline'ını döner; gerekiyorsa yükler."""
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        from pyannote.audio import Pipeline  # geç import

        if not config.HF_TOKEN:
            raise DiarizationError(
                "HF_TOKEN tanımlı değil. pyannote modeli için Hugging Face token'ı "
                "gereklidir. Lütfen .env dosyasına HF_TOKEN=hf_xxx ekleyin ve "
                "https://huggingface.co/pyannote/speaker-diarization-3.1 adresinden "
                "model kullanım koşullarını kabul edin."
            )

        device = _resolve_device(config.WHISPER_DEVICE)  # aynı cihaz tercihini paylaş
        logger.info(
            "pyannote '%s' pipeline'ı yükleniyor (cihaz: %s) …",
            config.PYANNOTE_MODEL,
            device,
        )
        try:
            from huggingface_hub import login
            login(token=config.HF_TOKEN)
            _pyannote_pipeline = Pipeline.from_pretrained(
                config.PYANNOTE_MODEL,
            )
            if _pyannote_pipeline is None:
                raise DiarizationError(
                    "Pipeline.from_pretrained() None döndürdü. "
                    "HF_TOKEN geçersiz olabilir veya "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1 "
                    "adresinde model kullanım koşulları kabul edilmemiş olabilir."
                )
            _pyannote_pipeline.to(torch.device(device))
        except DiarizationError:
            raise
        except Exception as exc:
            raise DiarizationError(f"pyannote pipeline yüklenemedi: {exc}") from exc
        logger.info("pyannote pipeline yüklendi.")
    return _pyannote_pipeline


# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def process(video_input: VideoInput) -> list[AnnotatedSegment]:
    """
    VideoInput → List[AnnotatedSegment]

    1. Whisper ile transkript al (veya existing_transcript kullan).
    2. pyannote ile diarisation yap.
    3. İkisini birleştir.

    Parameters
    ----------
    video_input:
        input_handler.py tarafından üretilen standart girdi nesnesi.

    Returns
    -------
    Kronolojik sırayla annotated segment listesi.
    """
    audio_path = Path(video_input.audio_path)
    if not audio_path.exists():
        raise AudioProcessorError(
            f"Ses dosyası bulunamadı: {audio_path}. "
            "input_handler'ın düzgün tamamlandığından emin olun."
        )

    logger.info("Ses işleme başlıyor: '%s'", video_input.title)

    # ── Adım 1: Transkript ────────────────────────────────────────────────────
    if video_input.existing_transcript:
        logger.info(
            "Mevcut YouTube altyazısı kullanılıyor (%d karakter); Whisper atlanıyor.",
            len(video_input.existing_transcript),
        )
        transcript_segments = _wrap_existing_transcript(
            video_input.existing_transcript,
            video_input.duration,
        )
    else:
        transcript_segments = _run_whisper(audio_path)

    logger.info("Transkript hazır — %d segment.", len(transcript_segments))

    # ── Adım 2: Diarisation ───────────────────────────────────────────────────
    speaker_segments = _run_diarization(audio_path)
    logger.info("Diarisation tamamlandı — %d konuşmacı bloğu.", len(speaker_segments))

    # ── Adım 3: Birleştirme ───────────────────────────────────────────────────
    annotated = _merge(transcript_segments, speaker_segments)
    logger.info(
        "Birleştirme tamamlandı — %d annotated segment üretildi.",
        len(annotated),
    )

    return annotated


# ── Adım 1: Whisper ───────────────────────────────────────────────────────────

def _run_whisper(audio_path: Path) -> list[TranscriptSegment]:
    """
    Whisper Large-v3 ile ses dosyasını transkript eder.

    - Dil otomatik algılanır (ilk 30 saniyelik probe).
    - Her segment için start/end timestamp döner.
    """
    model = _get_whisper()

    logger.info("Whisper transkripsiyon başlıyor: %s", audio_path.name)

    transcribe_kwargs: dict = {
        "verbose": False,
        "word_timestamps": False,   # segment-level timestamp yeterli
    }
    if config.WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = config.WHISPER_LANGUAGE
        logger.debug("Dil sabitlendi: %s", config.WHISPER_LANGUAGE)
    else:
        logger.debug("Dil otomatik algılanacak.")

    try:
        result = model.transcribe(str(audio_path), **transcribe_kwargs)
    except Exception as exc:
        raise WhisperError(f"Whisper transkripsiyon başarısız: {exc}") from exc

    detected_lang: str = result.get("language", "unknown")
    logger.info("Algılanan dil: %s", detected_lang)

    segments: list[TranscriptSegment] = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
                language=detected_lang,
            )
        )

    if not segments:
        logger.warning("Whisper hiç segment üretemedi. Ses dosyası sessiz olabilir.")

    return segments


def _wrap_existing_transcript(raw_text: str, total_duration: float) -> list[TranscriptSegment]:
    """
    YouTube altyazısı düz metin olarak gelir; timestamp yoktur.
    Metni cümlelere böler, cümleleri süreye eşit aralıklarla dağıtır.
    Bu yaklaşık timestamp'ler diarisation merge adımında düzeltilir.
    """
    # Nokta/soru/ünlem/satırsonu sınırlarından böl
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", raw_text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    step = total_duration / len(sentences)
    lang = config.WHISPER_LANGUAGE or "unknown"

    return [
        TranscriptSegment(
            start=round(i * step, 3),
            end=round((i + 1) * step, 3),
            text=sentence,
            language=lang,
        )
        for i, sentence in enumerate(sentences)
    ]


# ── Adım 2: Diarisation ───────────────────────────────────────────────────────

# pyannote konuşmacı etiketleri sıfır tabanlı sayı içerir (SPEAKER_00, SPEAKER_01…).
# Bunları okunabilir SPEAKER_A, SPEAKER_B… formatına dönüştürürüz.
_LETTER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _speaker_label(pyannote_label: str, mapping: dict[str, str]) -> str:
    """
    "SPEAKER_00" → "SPEAKER_A" gibi etiket dönüşümü.
    Yeni etiket gelince mapping'e eklenir; 26'dan fazlası için "SPEAKER_AA", … gibi devam eder.
    """
    if pyannote_label not in mapping:
        idx = len(mapping)
        if idx < 26:
            letter = _LETTER_LABELS[idx]
        else:
            letter = _LETTER_LABELS[idx // 26 - 1] + _LETTER_LABELS[idx % 26]
        mapping[pyannote_label] = f"SPEAKER_{letter}"
    return mapping[pyannote_label]


def _run_diarization(audio_path: Path) -> list[SpeakerSegment]:
    """
    pyannote.audio ile kim ne zaman konuştu tespiti yapar.
    """
    pipeline = _get_pyannote()

    logger.info("Diarisation başlıyor: %s", audio_path.name)

    diarize_kwargs: dict = {}
    if config.MIN_SPEAKERS > 1 or config.MAX_SPEAKERS < 10:
        diarize_kwargs["min_speakers"] = config.MIN_SPEAKERS
        diarize_kwargs["max_speakers"] = config.MAX_SPEAKERS
        logger.debug(
            "Konuşmacı sayısı aralığı: %d – %d",
            config.MIN_SPEAKERS,
            config.MAX_SPEAKERS,
        )

    try:
        diarization = pipeline(str(audio_path), **diarize_kwargs)
    except Exception as exc:
        raise DiarizationError(f"pyannote diarisation başarısız: {exc}") from exc

    label_map: dict[str, str] = {}
    segments: list[SpeakerSegment] = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker=_speaker_label(speaker, label_map),
            )
        )

    unique_speakers = sorted({s.speaker for s in segments})
    logger.info(
        "Tespit edilen konuşmacılar (%d): %s",
        len(unique_speakers),
        ", ".join(unique_speakers),
    )

    return segments


# ── Adım 3: Birleştirme ───────────────────────────────────────────────────────

_UNKNOWN_SPEAKER = "SPEAKER_?"


def _overlap(
    t_start: float, t_end: float,
    s_start: float, s_end: float,
) -> float:
    """İki zaman aralığının örtüşme süresini saniye olarak döner."""
    return max(0.0, min(t_end, s_end) - max(t_start, s_start))


def _best_speaker(
    t_start: float,
    t_end: float,
    speaker_segments: list[SpeakerSegment],
) -> str:
    """
    Bir TranscriptSegment zaman aralığıyla en çok örtüşen konuşmacıyı bulur.
    Eşitlik durumunda alfabetik olarak küçük olan seçilir (deterministik).
    """
    best_speaker = _UNKNOWN_SPEAKER
    best_overlap = 0.0

    for sp in speaker_segments:
        # Zaman aralığı dışındaki segment'leri erkenden atla (performans)
        if sp.end < t_start or sp.start > t_end:
            continue
        ov = _overlap(t_start, t_end, sp.start, sp.end)
        if ov > best_overlap or (ov == best_overlap and sp.speaker < best_speaker):
            best_overlap = ov
            best_speaker = sp.speaker

    return best_speaker


def _merge(
    transcript: list[TranscriptSegment],
    speakers: list[SpeakerSegment],
) -> list[AnnotatedSegment]:
    """
    Her TranscriptSegment'e en çok örtüşen SpeakerSegment'i atar.

    Konuşmacı bulunamazsa (sessiz bölge, diarisation boşluğu) SPEAKER_? kullanılır.
    Sonuç kronolojik sıraya göre döner.
    """
    if not transcript:
        logger.warning("Transkript boş; birleştirme atlanıyor.")
        return []

    if not speakers:
        logger.warning(
            "Diarisation sonucu boş; tüm segmentler '%s' olarak işaretleniyor.",
            _UNKNOWN_SPEAKER,
        )
        return [
            AnnotatedSegment(
                start=s.start,
                end=s.end,
                text=s.text,
                speaker=_UNKNOWN_SPEAKER,
                language=s.language,
            )
            for s in sorted(transcript, key=lambda x: x.start)
        ]

    annotated: list[AnnotatedSegment] = []
    unknown_count = 0

    for seg in sorted(transcript, key=lambda x: x.start):
        speaker = _best_speaker(seg.start, seg.end, speakers)
        if speaker == _UNKNOWN_SPEAKER:
            unknown_count += 1
        annotated.append(
            AnnotatedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=speaker,
                language=seg.language,
            )
        )

    if unknown_count:
        logger.warning(
            "%d segment için konuşmacı ataması yapılamadı ('%s' olarak işaretlendi).",
            unknown_count,
            _UNKNOWN_SPEAKER,
        )

    return annotated


# ── Yardımcı: önbelleği temizle (test / yeniden yükleme için) ─────────────────

def unload_models() -> None:
    """
    Yüklü modelleri bellekten siler.
    Bellek kısıtlı ortamlarda pipeline adımları arasında çağrılabilir.
    """
    global _whisper_model, _pyannote_pipeline
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        logger.info("Whisper modeli bellekten kaldırıldı.")
    if _pyannote_pipeline is not None:
        del _pyannote_pipeline
        _pyannote_pipeline = None
        logger.info("pyannote pipeline bellekten kaldırıldı.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache temizlendi.")
