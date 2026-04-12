"""
note_generator.py
─────────────────
merger.py'dan gelen List[MergedItem] alır;
Mistral 7B-Instruct ile yapılandırılmış toplantı notu üretir.

Akış:
  List[MergedItem]
      → metin bloklarına chunk'la
      → her chunk için LLM çağrısı → JSON çıktı
      → chunk sonuçlarını birleştir
      → MeetingNotes dataclass döner

Model lazy-load edilir; unload_models() ile VRAM'den temizlenebilir.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

import torch

import config
from pipeline.merger import MergedItem

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("note_generator")

# ── Sabitler ──────────────────────────────────────────────────────────────────

_WORDS_PER_TOKEN    = 1.3     # kaba token tahmini: kelime sayısı × bu çarpan
_CONTENT_TYPE_ICONS = {       # chunk metninde görsel ayraç
    "terminal": "💻",
    "code":     "🖥️",
    "slide":    "📊",
    "diagram":  "📈",
    "ui":       "🖱️",
}
_DEFAULT_ICON = "📸"

# LLM'den beklenen JSON şeması (prompt'a eklemek için)
_JSON_SCHEMA = """\
{
  "summary": "<3-5 cümle genel özet>",
  "decisions": ["<karar 1>", "<karar 2>"],
  "action_items": [
    {"owner": "<kişi>", "task": "<yapılacak iş>", "due": "<tarih veya null>"}
  ],
  "technical_moments": [
    {"timestamp": <saniye float>, "description": "<teknik detay>"}
  ]
}"""

# ── Veri modelleri ────────────────────────────────────────────────────────────

@dataclass
class ActionItem:
    owner: str
    task:  str
    due:   str | None = None


@dataclass
class TechnicalMoment:
    timestamp:  float
    description: str
    frame_path:  str | None = None


@dataclass
class MeetingNotes:
    title:              str
    date:               str
    duration_str:       str
    speakers:           list[str]
    summary:            str
    transcript:         list[MergedItem]   # orijinal kronolojik akış — PDF için
    decisions:          list[str]          = field(default_factory=list)
    action_items:       list[ActionItem]   = field(default_factory=list)
    technical_moments:  list[TechnicalMoment] = field(default_factory=list)
    raw_llm_fallback:   list[str]          = field(default_factory=list)
                                           # JSON parse edilemeyen chunk yanıtları


# ── Hata sınıfları ────────────────────────────────────────────────────────────

class NoteGeneratorError(Exception):
    """Genel not üretme hatası."""


class LLMLoadError(NoteGeneratorError):
    """Mistral modeli veya tokenizer yüklenemedi."""


# ── Lazy-load model önbelleği ─────────────────────────────────────────────────

_llm_model     = None   # MistralForCausalLM (veya AutoModelForCausalLM)
_llm_tokenizer = None   # AutoTokenizer


def _get_llm() -> tuple:
    """Mistral modelini ve tokenizer'ı döner; gerekiyorsa yükler."""
    global _llm_model, _llm_tokenizer

    if _llm_model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(
            "Mistral modeli yükleniyor: '%s' (4-bit=%s) …",
            config.LLM_MODEL_NAME,
            config.LLM_LOAD_IN_4BIT,
        )

        model_kwargs: dict = {
            "device_map":  config.LLM_DEVICE_MAP,
            "torch_dtype": torch.float16,
        }

        if config.LLM_LOAD_IN_4BIT:
            if not torch.cuda.is_available():
                logger.warning(
                    "4-bit quantization CUDA gerektiriyor ancak CUDA bulunamadı; "
                    "quantization devre dışı bırakılıyor."
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

        try:
            _llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
            _llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME, **model_kwargs
            )
            _llm_model.eval()
        except Exception as exc:
            raise LLMLoadError(f"Mistral modeli yüklenemedi: {exc}") from exc

        logger.info("Mistral modeli yüklendi.")

    return _llm_model, _llm_tokenizer


# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def generate(
    items: list[MergedItem],
    title: str,
    duration: float,
    date: str | None = None,
) -> MeetingNotes:
    """
    List[MergedItem] → MeetingNotes

    Parameters
    ----------
    items:
        merger.merge() çıktısı — kronolojik sıralı.
    title:
        Toplantı başlığı (VideoInput.title'dan gelir).
    duration:
        Toplantı süresi saniye cinsinden.
    date:
        Toplantı tarihi (YYYY-MM-DD). None ise bugünün tarihi kullanılır.

    Returns
    -------
    Yapılandırılmış MeetingNotes nesnesi.
    """
    if not items:
        logger.warning("Boş MergedItem listesi geldi; minimal not üretiliyor.")
        return _empty_notes(title, duration, date)

    date_str     = date or datetime.now().strftime("%Y-%m-%d")
    duration_str = _fmt_duration(duration)
    speakers     = _extract_speakers(items)

    logger.info(
        "Not üretimi başlıyor — '%s', %s, %d konuşmacı, %d item.",
        title, duration_str, len(speakers), len(items),
    )

    # ── Model yükle ───────────────────────────────────────────────────────────
    _get_llm()

    # ── Chunk'lara böl ve LLM çağrıları ──────────────────────────────────────
    chunks = _split_into_chunks(items)
    logger.info("%d item %d chunk'a bölündü.", len(items), len(chunks))

    all_summaries:    list[str]          = []
    all_decisions:    list[str]          = []
    all_action_items: list[ActionItem]   = []
    all_tech_moments: list[TechnicalMoment] = []
    raw_fallbacks:    list[str]          = []

    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Chunk işleniyor: %d / %d", idx, len(chunks))
        chunk_text = _render_chunk(chunk)
        parsed     = _call_llm(chunk_text, idx, len(chunks))

        if parsed is None:
            # JSON çözümlenemedi; ham yanıtı fallback olarak sakla
            continue

        if parsed.get("_raw_fallback"):
            raw_fallbacks.append(parsed["_raw_fallback"])
            continue

        if parsed.get("summary"):
            all_summaries.append(parsed["summary"])

        for d in parsed.get("decisions", []):
            if isinstance(d, str) and d.strip():
                all_decisions.append(d.strip())

        for ai in parsed.get("action_items", []):
            item = _parse_action_item(ai)
            if item:
                all_action_items.append(item)

        for tm in parsed.get("technical_moments", []):
            moment = _parse_technical_moment(tm, chunk)
            if moment:
                all_tech_moments.append(moment)

    # ── Sonuçları birleştir ───────────────────────────────────────────────────
    final_summary = _merge_summaries(all_summaries, title)
    deduped_decisions = _deduplicate(all_decisions)

    logger.info(
        "Not üretimi tamamlandı — özet: %d cümle, karar: %d, "
        "action item: %d, teknik an: %d.",
        len(final_summary.split(". ")),
        len(deduped_decisions),
        len(all_action_items),
        len(all_tech_moments),
    )

    return MeetingNotes(
        title=title,
        date=date_str,
        duration_str=duration_str,
        speakers=speakers,
        summary=final_summary,
        transcript=items,
        decisions=deduped_decisions,
        action_items=all_action_items,
        technical_moments=sorted(all_tech_moments, key=lambda x: x.timestamp),
        raw_llm_fallback=raw_fallbacks,
    )


# ── Chunk oluşturma ───────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Kelime sayısına dayalı kaba token tahmini."""
    return int(len(text.split()) * _WORDS_PER_TOKEN)


def _split_into_chunks(items: list[MergedItem]) -> list[list[MergedItem]]:
    """
    MergedItem listesini LLM context window'una sığacak parçalara böler.

    - Her item'ın render edilmiş metin boyutuna bakılır.
    - Chunk sınırı speech/frame ortasında değil, item sınırlarında yapılır.
    - Tek bir item'ın kendisi limiti aşarsa (çok uzun terminal çıktısı) yine de
      tek başına bir chunk oluşturur — truncate etmiyoruz.
    """
    chunks:       list[list[MergedItem]] = []
    current:      list[MergedItem]       = []
    current_tokens = 0

    for item in items:
        item_text   = _render_item(item)
        item_tokens = _estimate_tokens(item_text)

        if current and (current_tokens + item_tokens) > config.LLM_MAX_CHUNK_TOKENS:
            chunks.append(current)
            current       = []
            current_tokens = 0

        current.append(item)
        current_tokens += item_tokens

    if current:
        chunks.append(current)

    return chunks


# ── Metin render ──────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    """123.4 → '[02:03]' formatı."""
    total = int(seconds)
    m, s  = divmod(total, 60)
    return f"[{m:02d}:{s:02d}]"


def _render_item(item: MergedItem) -> str:
    """Tek bir MergedItem'ı LLM'e gönderilecek satır formatına çevirir."""
    ts = _fmt_ts(item.timestamp)
    if item.item_type == "speech":
        speaker = item.speaker or "SPEAKER_?"
        text    = item.text or ""
        return f'{ts} {speaker}: "{text}"'
    else:
        icon = _CONTENT_TYPE_ICONS.get(item.content_type or "", _DEFAULT_ICON)
        ctype = item.content_type or "frame"
        desc  = item.description or ""
        if item.extracted_text:
            return f"{ts} {icon} {ctype}:\n{item.extracted_text}\n({desc})"
        return f"{ts} {icon} {ctype}: {desc}"


def _render_chunk(chunk: list[MergedItem]) -> str:
    """Chunk'taki tüm item'ları tek bir metin bloğuna dönüştürür."""
    return "\n".join(_render_item(item) for item in chunk)


# ── LLM çağrısı ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = config.NOTE_SYSTEM_PROMPT

_USER_TEMPLATE = """\
Aşağıdaki toplantı transkriptini analiz et ve sadece JSON formatında yanıt ver.
Başka hiçbir metin ekleme. Sadece geçerli JSON döndür.

Beklenen JSON şeması:
{schema}

Transkript:
{transcript}
"""


@torch.inference_mode()
def _call_llm(chunk_text: str, chunk_idx: int, total_chunks: int) -> dict | None:
    """
    Tek bir chunk metnini Mistral'a gönderir, JSON dict döner.

    Dönüş değerleri:
      dict       → başarılı JSON parse
      {"_raw_fallback": str} → parse başarısız ama yanıt var
      None       → model hatası (üretim tamamen başarısız)
    """
    model, tokenizer = _get_llm()

    # Mistral chat template: [INST] ... [/INST]
    user_content = _USER_TEMPLATE.format(
        schema=_JSON_SCHEMA,
        transcript=chunk_text,
    )

    messages = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Bazı tokenizer versiyonları system mesajını desteklemez;
        # system içeriğini user mesajına önek olarak ekle.
        combined = f"{_SYSTEM_PROMPT}\n\n{user_content}"
        messages_fallback = [{"role": "user", "content": combined}]
        prompt = tokenizer.apply_chat_template(
            messages_fallback,
            tokenize=False,
            add_generation_prompt=True,
        )

    device = next(model.parameters()).device

    try:
        inputs     = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
            do_sample=config.LLM_TEMPERATURE > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    except Exception as exc:
        logger.error("Chunk %d/%d LLM üretimi başarısız: %s", chunk_idx, total_chunks, exc)
        return None

    input_len  = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    return _parse_json_response(raw_output, chunk_idx, total_chunks)


def _parse_json_response(raw: str, chunk_idx: int, total_chunks: int) -> dict:
    """
    LLM çıktısından JSON nesnesini çıkarır.

    1. Doğrudan JSON parse dene.
    2. Başarısız olursa ```json ... ``` bloğu veya { ... } bloğu ara.
    3. Yine başarısız olursa {"_raw_fallback": raw} döner.
    """
    # Deneme 1: doğrudan
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Deneme 2: ```json ... ``` veya ``` ... ``` bloğu
    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if block_match:
        try:
            return json.loads(block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Deneme 3: ilk { ... } bloğunu bul
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(
        "Chunk %d/%d JSON parse edilemedi; ham metin kaydediliyor.",
        chunk_idx,
        total_chunks,
    )
    return {"_raw_fallback": raw}


# ── Yardımcı ayrıştırıcılar ───────────────────────────────────────────────────

def _parse_action_item(data: dict) -> ActionItem | None:
    if not isinstance(data, dict):
        return None
    task = (data.get("task") or "").strip()
    if not task:
        return None
    return ActionItem(
        owner=str(data.get("owner") or "Belirsiz").strip(),
        task=task,
        due=str(data["due"]).strip() if data.get("due") else None,
    )


def _parse_technical_moment(data: dict, chunk: list[MergedItem]) -> TechnicalMoment | None:
    if not isinstance(data, dict):
        return None
    desc = (data.get("description") or "").strip()
    if not desc:
        return None

    # Timestamp al; LLM bazen string döndürebilir
    try:
        ts = float(data.get("timestamp") or 0.0)
    except (ValueError, TypeError):
        ts = chunk[0].timestamp if chunk else 0.0

    # Aynı timestamp'e en yakın frame_path bul
    frame_path = _nearest_frame_path(ts, chunk)

    return TechnicalMoment(timestamp=ts, description=desc, frame_path=frame_path)


def _nearest_frame_path(ts: float, items: list[MergedItem]) -> str | None:
    """Verilen timestamp'e en yakın frame item'ının frame_path'ini döner."""
    best_path: str | None = None
    best_dist: float      = float("inf")
    for item in items:
        if item.item_type == "frame" and item.frame_path:
            dist = abs(item.timestamp - ts)
            if dist < best_dist:
                best_dist = dist
                best_path = item.frame_path
    return best_path


# ── Sonuç birleştirme ─────────────────────────────────────────────────────────

def _merge_summaries(summaries: list[str], title: str) -> str:
    """
    Birden fazla chunk özeti varsa LLM ile tek özete indirir.
    Tek özet varsa direkt döner. Hiç yoksa fallback metin üretir.
    """
    if not summaries:
        return f"'{title}' toplantısına ait özet üretilemedi."

    if len(summaries) == 1:
        return summaries[0]

    logger.info(
        "%d chunk özeti tek özete birleştiriliyor …", len(summaries)
    )

    combined = "\n\n".join(
        f"Bölüm {i + 1}: {s}" for i, s in enumerate(summaries)
    )
    merge_prompt = (
        "Aşağıdaki toplantı bölümlerinin özetlerini, "
        "tekrarları çıkararak tek bir akıcı Türkçe özete dönüştür (3-5 cümle):\n\n"
        + combined
    )

    try:
        model, tokenizer = _get_llm()
        device   = next(model.parameters()).device
        messages = [{"role": "user", "content": merge_prompt}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs   = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        merged    = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        if merged:
            return merged
    except Exception as exc:
        logger.warning("Özet birleştirme başarısız, özetler birleştiriliyor: %s", exc)

    # Fallback: özetleri sadece birleştir
    return " ".join(summaries)


def _deduplicate(items: list[str]) -> list[str]:
    """Listeden yinelenen öğeleri sıra koruyarak kaldırır."""
    seen:   set[str]  = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ── Küçük yardımcılar ─────────────────────────────────────────────────────────

def _extract_speakers(items: list[MergedItem]) -> list[str]:
    """MergedItem listesinden benzersiz konuşmacıları alfabetik sırayla döner."""
    return sorted({
        item.speaker
        for item in items
        if item.item_type == "speech" and item.speaker and item.speaker != "SPEAKER_?"
    })


def _fmt_duration(seconds: float) -> str:
    """3661.0 → '1s 1d 01sn' formatı."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s         = divmod(remainder, 60)
    if h:
        return f"{h}s {m}d {s:02d}sn"
    if m:
        return f"{m}d {s:02d}sn"
    return f"{s}sn"


def _empty_notes(title: str, duration: float, date: str | None) -> MeetingNotes:
    return MeetingNotes(
        title=title,
        date=date or datetime.now().strftime("%Y-%m-%d"),
        duration_str=_fmt_duration(duration),
        speakers=[],
        summary="Transkript boş; not üretilemedi.",
        transcript=[],
    )


# ── Model temizleme ───────────────────────────────────────────────────────────

def unload_models() -> None:
    """
    Mistral modelini ve tokenizer'ı bellekten siler.
    Pipeline tamamlandıktan sonra çağrılabilir.
    """
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        del _llm_model
        del _llm_tokenizer
        _llm_model = _llm_tokenizer = None
        logger.info("Mistral modeli bellekten kaldırıldı.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache temizlendi.")
