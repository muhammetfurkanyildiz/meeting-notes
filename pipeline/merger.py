"""
merger.py
─────────
audio_processor.py  →  List[AnnotatedSegment]   (konuşma)
vlm_processor.py    →  List[VLMResult]           (frame)

İkisini timestamp üzerinden kronolojik sıraya dizer ve
aynı konuşmacının ardışık speech segmentlerini birleştirir.

Çıktı: List[MergedItem]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pipeline.audio_processor import AnnotatedSegment
from pipeline.vlm_processor import VLMResult

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("merger")

# ── Sabitler ──────────────────────────────────────────────────────────────────

# Aynı konuşmacının ardışık segmentleri arasındaki maksimum boşluk (saniye).
# Bu değerden küçük boşluklarda iki segment tek segmente birleştirilir.
_SPEECH_MERGE_GAP_SEC = 1.0

# ── Veri modeli ───────────────────────────────────────────────────────────────

@dataclass
class MergedItem:
    """
    Pipeline'ın birleşik kronolojik birimi.
    item_type == "speech" → speaker/text/language dolu, frame alanları None.
    item_type == "frame"  → frame alanları dolu, speech alanları None.
    """
    timestamp:      float
    item_type:      str          # "speech" | "frame"

    # Speech alanları
    speaker:        str | None = None
    text:           str | None = None
    language:       str | None = None
    end_timestamp:  float | None = None  # speech segmentinin bitiş zamanı

    # Frame alanları
    frame_path:     str | None = None
    description:    str | None = None
    extracted_text: str | None = None
    content_type:   str | None = None


# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def merge(
    segments: list[AnnotatedSegment],
    vlm_results: list[VLMResult],
) -> list[MergedItem]:
    """
    AnnotatedSegment + VLMResult listelerini alır,
    timestamp'e göre sıralar ve birleşik MergedItem listesi döner.

    Adımlar:
      1. AnnotatedSegment → MergedItem("speech")
      2. VLMResult        → MergedItem("frame")
      3. Tümünü timestamp'e göre sırala
      4. Aynı konuşmacının ardışık speech segmentlerini birleştir

    Parameters
    ----------
    segments:
        audio_processor.process() çıktısı.
    vlm_results:
        vlm_processor.process() çıktısı. Boş olabilir (ses-only toplantı).

    Returns
    -------
    Kronolojik sırayla MergedItem listesi.
    """
    speech_items = _segments_to_items(segments)
    frame_items  = _vlm_to_items(vlm_results)

    logger.info(
        "Birleştirme başlıyor — speech: %d segment, frame: %d sonuç.",
        len(speech_items),
        len(frame_items),
    )

    # ── Kronolojik sıralama ───────────────────────────────────────────────────
    # Frame item'ları timestamp noktasına sahip; speech item'ları start zamanını taşır.
    # Aynı timestamp'de speech ve frame çakışırsa speech önce gelsin (daha anlamlı bağlam).
    combined = sorted(
        speech_items + frame_items,
        key=lambda item: (item.timestamp, 0 if item.item_type == "speech" else 1),
    )

    # ── Speech segmentlerini birleştir ────────────────────────────────────────
    merged = _merge_speech_gaps(combined)

    n_speech = sum(1 for x in merged if x.item_type == "speech")
    n_frame  = sum(1 for x in merged if x.item_type == "frame")

    logger.info(
        "Birleştirme tamamlandı — toplam %d item (speech: %d, frame: %d).",
        len(merged),
        n_speech,
        n_frame,
    )

    return merged


# ── Dönüşüm yardımcıları ──────────────────────────────────────────────────────

def _segments_to_items(segments: list[AnnotatedSegment]) -> list[MergedItem]:
    """AnnotatedSegment listesini speech MergedItem listesine çevirir."""
    items: list[MergedItem] = []
    for seg in segments:
        items.append(MergedItem(
            timestamp=seg.start,
            item_type="speech",
            speaker=seg.speaker,
            text=seg.text.strip() if seg.text else "",
            language=seg.language,
            end_timestamp=seg.end,
        ))
    return items


def _vlm_to_items(vlm_results: list[VLMResult]) -> list[MergedItem]:
    """VLMResult listesini frame MergedItem listesine çevirir."""
    items: list[MergedItem] = []
    for res in vlm_results:
        items.append(MergedItem(
            timestamp=res.timestamp,
            item_type="frame",
            frame_path=res.frame_path,
            description=res.description,
            extracted_text=res.extracted_text,
            content_type=res.content_type,
        ))
    return items


# ── Speech birleştirme mantığı ────────────────────────────────────────────────

def _merge_speech_gaps(items: list[MergedItem]) -> list[MergedItem]:
    """
    Kronolojik sıralı item listesinde ardışık speech segmentlerini birleştirir.

    Birleştirme koşulları (hepsi aynı anda sağlanmalı):
      - Her iki item de item_type == "speech"
      - Her iki item de aynı speaker
      - İkinci segmentin başlangıcı ile birincinin bitişi arasındaki boşluk
        _SPEECH_MERGE_GAP_SEC saniyeden az

    Aradaki frame item'ları birleştirmeyi keser — konuşmacı ekran değiştirdikten
    sonra aynı cümleye devam ediyor olsa da o iki segment ayrı kalır.
    """
    if not items:
        return []

    result: list[MergedItem] = []
    current = items[0]
    merged_count = 0

    for next_item in items[1:]:

        # Birleştirme sadece ardışık iki speech item arasında olur
        if (
            current.item_type == "speech"
            and next_item.item_type == "speech"
            and current.speaker == next_item.speaker
            and _gap(current, next_item) < _SPEECH_MERGE_GAP_SEC
        ):
            # Mevcut segmenti genişlet
            merged_text = _join_text(current.text, next_item.text)
            current = MergedItem(
                timestamp=current.timestamp,
                item_type="speech",
                speaker=current.speaker,
                text=merged_text,
                language=current.language,
                end_timestamp=next_item.end_timestamp,
            )
            merged_count += 1

        else:
            result.append(current)
            current = next_item

    result.append(current)  # son item'ı da ekle

    if merged_count:
        logger.debug(
            "%d ardışık speech segmenti birleştirildi (boşluk < %.1f sn).",
            merged_count,
            _SPEECH_MERGE_GAP_SEC,
        )

    return result


# ── Küçük yardımcılar ─────────────────────────────────────────────────────────

def _gap(current: MergedItem, next_item: MergedItem) -> float:
    """
    İki ardışık speech item arasındaki sessizlik süresini döner.
    current.end_timestamp None ise gap sonsuz kabul edilir (birleştirme engellenir).
    """
    if current.end_timestamp is None:
        return float("inf")
    return max(0.0, next_item.timestamp - current.end_timestamp)


def _join_text(a: str | None, b: str | None) -> str:
    """
    İki metin parçasını birleştirir.
    Birinci parça nokta/soru/ünlem ile bitiyorsa boşluk, bitmiyorsa virgül+boşluk ekler.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    # Birinci parça zaten cümle sonu işareti taşıyorsa düz birleştir
    if a[-1] in ".!?…":
        return f"{a} {b}"
    return f"{a}, {b}"
