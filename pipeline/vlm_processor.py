"""
vlm_processor.py
────────────────
frame_processor.py'dan gelen List[FrameResult] alır;
her frame'i Qwen2.5-VL-7B-Instruct ile analiz eder ve
VLMResult listesi döner.

Akış:
  FrameResult  →  Qwen2.5-VL  →  VLMResult
                  (description + content_type + extracted_text)

Model lazy-load edilir; 4-bit quantization config ile kontrol edilir.
Frame'ler teker teker işlenir — hata varsa o frame loglanıp atlanır.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

import config
from pipeline.frame_processor import FrameResult

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("vlm_processor")

# ── Sabitler: içerik tipi keyword'leri ───────────────────────────────────────

# Her kural: (content_type, [eşleşecek alt diziler])
# Sıralama önemli: daha spesifik kurallar önce gelir.
_CONTENT_RULES: list[tuple[str, list[str]]] = [
    ("terminal", ["terminal", "komut satırı", "command line", "$ ", ">>> ", "bash", "zsh", "powershell", "cmd"]),
    ("code",     ["kod", "def ", "class ", "function", "import ", "return ", "#!/", "```"]),
    ("slide",    ["slayt", "sunum", "başlık", "madde işareti", "bullet", "slide"]),
    ("diagram",  ["grafik", "tablo", "diyagram", "chart", "diagram", "flowchart", "şema", "akış"]),
]
_FALLBACK_TYPE = "ui"

# ── Veri modeli ───────────────────────────────────────────────────────────────

@dataclass
class VLMResult:
    """Tek bir frame'in VLM analiz çıktısı."""
    timestamp:      float
    frame_path:     str         # PDF'de görüntü olarak kullanılacak
    description:    str         # VLM'in serbest metin açıklaması
    extracted_text: str | None  # terminal / kod içeriği (ham, aynen)
    content_type:   str         # terminal | code | slide | diagram | ui | unknown


# ── Hata sınıfları ────────────────────────────────────────────────────────────

class VLMProcessorError(Exception):
    """Genel VLM işleme hatası."""


class VLMLoadError(VLMProcessorError):
    """Model veya processor yüklenemedi."""


# ── Lazy-load model önbelleği ─────────────────────────────────────────────────

_vlm_model     = None   # Qwen2.5VLForConditionalGeneration
_vlm_processor = None   # AutoProcessor


def _get_vlm() -> tuple:
    """VLM modelini ve processor'ı döner; gerekiyorsa yükler."""
    global _vlm_model, _vlm_processor

    if _vlm_model is None:
        # Geç import: transformers ağır, sadece bu modül çağrıldığında yüklenir
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )

        logger.info(
            "Qwen2.5-VL modeli yükleniyor: '%s' (4-bit=%s) …",
            config.VLM_MODEL_NAME,
            config.VLM_LOAD_IN_4BIT,
        )

        model_kwargs: dict = {
            "device_map": config.VLM_DEVICE_MAP,
            "torch_dtype": torch.float16,
        }

        if config.VLM_LOAD_IN_4BIT:
            if not torch.cuda.is_available():
                logger.warning(
                    "4-bit quantization CUDA gerektiriyor; CUDA bulunamadı. "
                    "Quantization devre dışı bırakılıyor."
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

        try:
            _vlm_model = AutoModelForImageTextToText.from_pretrained(
                config.VLM_MODEL_NAME,
                **model_kwargs,
            )
            _vlm_model.eval()

            _vlm_processor = AutoProcessor.from_pretrained(config.VLM_MODEL_NAME)
        except Exception as exc:
            raise VLMLoadError(f"Qwen2.5-VL yüklenemedi: {exc}") from exc

        logger.info("Qwen2.5-VL modeli yüklendi.")

    return _vlm_model, _vlm_processor


# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def process(frame_results: list[FrameResult]) -> list[VLMResult]:
    """
    List[FrameResult] → List[VLMResult]

    Her frame Qwen2.5-VL ile analiz edilir.
    Hata veren frame'ler atlanır; tamamlanabildiği kadar sonuç döner.

    Parameters
    ----------
    frame_results:
        frame_processor.process() çıktısı.

    Returns
    -------
    Kronolojik sırayla VLMResult listesi.
    """
    if not frame_results:
        logger.warning("İşlenecek frame bulunamadı; VLM adımı atlanıyor.")
        return []

    total   = len(frame_results)
    results: list[VLMResult] = []
    errors  = 0

    logger.info("VLM analizi başlıyor — toplam %d frame.", total)

    # Model + processor'ı döngüye girmeden yükle (ilk gecikme burada olsun)
    _get_vlm()

    for idx, frame in enumerate(frame_results, start=1):

        # Her 10 frame'de bir ilerleme logu
        if idx == 1 or idx % 10 == 0 or idx == total:
            logger.info("VLM işleniyor: %d / %d", idx, total)

        result = _process_single(frame)
        if result is None:
            errors += 1
            continue
        results.append(result)

    logger.info(
        "VLM analizi tamamlandı — başarılı: %d, hatalı/atlanan: %d.",
        len(results),
        errors,
    )
    return results


# ── Tek frame işleme ──────────────────────────────────────────────────────────

def _process_single(frame: FrameResult) -> VLMResult | None:
    """
    Tek bir FrameResult'ı işler.
    Herhangi bir hata durumunda None döner (caller frame'i atlar).
    """
    frame_path = Path(frame.frame_path)
    if not frame_path.exists():
        logger.warning(
            "[%.2fs] Frame dosyası bulunamadı, atlanıyor: %s",
            frame.timestamp,
            frame_path,
        )
        return None

    try:
        image = Image.open(frame_path).convert("RGB")
    except Exception as exc:
        logger.warning(
            "[%.2fs] Görüntü açılamadı, atlanıyor: %s — %s",
            frame.timestamp,
            frame_path.name,
            exc,
        )
        return None

    try:
        description = _run_inference(image)
    except Exception as exc:
        logger.warning(
            "[%.2fs] VLM çıkarımı başarısız, atlanıyor: %s — %s",
            frame.timestamp,
            frame_path.name,
            exc,
        )
        return None

    content_type   = _detect_content_type(description)
    extracted_text = _extract_text(description, content_type)

    logger.debug(
        "[%.2fs] %s — %s",
        frame.timestamp,
        content_type.upper(),
        description[:80].replace("\n", " "),
    )

    return VLMResult(
        timestamp=frame.timestamp,
        frame_path=frame.frame_path,
        description=description,
        extracted_text=extracted_text,
        content_type=content_type,
    )


# ── VLM çıkarımı ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def _run_inference(image: Image.Image) -> str:
    """
    Tek bir PIL görüntüsünü Qwen2.5-VL'e gönderir, yanıt metnini döner.
    """
    from qwen_vl_utils import process_vision_info  # geç import

    model, processor = _get_vlm()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": config.VLM_PROMPT},
            ],
        }
    ]

    # Qwen2.5-VL chat template
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=config.VLM_MAX_NEW_TOKENS,
        do_sample=False,        # greedy; tutarlılık için
    )

    # Sadece yeni üretilen token'ları decode et (prompt kısmını at)
    input_len     = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response      = processor.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


# ── İçerik tipi tespiti ───────────────────────────────────────────────────────

def _detect_content_type(description: str) -> str:
    """
    VLM yanıtındaki keyword'lere göre içerik tipini belirler.
    Kurallar sırayla denenir; ilk eşleşen döner.
    Hiç eşleşme yoksa _FALLBACK_TYPE döner.
    """
    lower = description.lower()
    for content_type, keywords in _CONTENT_RULES:
        if any(kw in lower for kw in keywords):
            return content_type
    return _FALLBACK_TYPE


# ── Ham metin çıkarma ─────────────────────────────────────────────────────────

# Terminal/kod içeriklerinde VLM genellikle backtick bloğu veya
# "Komut: ..." / "Çıktı: ..." satırları üretir.
# Bunları extracted_text olarak ayırırız; diğer tipler için None döner.

_CODE_BLOCK_RE = re.compile(r"```[a-z]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def _extract_text(description: str, content_type: str) -> str | None:
    """
    terminal veya code tiplerinde ham metni döner.
    Önce Markdown code block arar; yoksa description'ın tamamını döner.
    Diğer tipler için None döner.
    """
    if content_type not in ("terminal", "code"):
        return None

    # Backtick bloklarını topla
    blocks = _CODE_BLOCK_RE.findall(description)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())

    # Satır içi backtick'ler
    inline = _INLINE_CODE_RE.findall(description)
    if inline:
        return "\n".join(inline)

    # Blok yoksa description'ın kendisi ham metin kabul edilir
    return description.strip()


# ── Model temizleme ───────────────────────────────────────────────────────────

def unload_models() -> None:
    """
    VLM modelini ve processor'ı bellekten siler.
    PARALLEL_AUDIO_VIDEO=False iken pipeline adımları arasında çağrılabilir.
    """
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        del _vlm_model
        del _vlm_processor
        _vlm_model = _vlm_processor = None
        logger.info("Qwen2.5-VL modeli bellekten kaldırıldı.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache temizlendi.")
