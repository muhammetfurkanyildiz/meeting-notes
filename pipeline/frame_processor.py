"""
frame_processor.py
──────────────────
VideoInput alır, video dosyasından frame'leri örnekler ve üç süzgeçten geçirir:

  Süzgeç 1 — CLIP benzerlik kontrolü  : önceki frame ile aynıysa atla
  Süzgeç 2 — Boş/avatar ekran tespiti : bilgisiz ekranları ele
  Süzgeç 3 — İçerik varlığı kontrolü  : asıl bilgi içeriği yoksa atla

Geçen frame'ler PNG olarak workdir/ altına kaydedilir ve FrameResult listesi döner.
CLIP modeli lazy-load edilir; unload_models() ile VRAM'den temizlenebilir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

import config
from pipeline.input_handler import VideoInput

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("frame_processor")

# ── Sabitler: süzgeç prompt'ları ─────────────────────────────────────────────

_EMPTY_PROMPTS: list[str] = [
    "a blank screen",
    "a black screen",
    "avatar initials on colored background",
    "video call with no screen share",
    "person's name on solid color",
]

_CONTENT_PROMPTS: list[str] = [
    "a computer screen with application or interface",
    "terminal or command line output",
    "presentation slide or document",
    "diagram or chart or table",
    "web browser or dashboard",
]

# ── Veri modeli ───────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Süzgeçlerden geçen tek bir frame'in çıktısı."""
    timestamp: float        # videonun kaçıncı saniyesinden alındı
    frame_path: str         # kaydedilen PNG dosyasının mutlak yolu
    clip_scores: dict       # debug: {"similarity": float, "empty_max": float, "content_max": float}


# ── Hata sınıfları ────────────────────────────────────────────────────────────

class FrameProcessorError(Exception):
    """Genel frame işleme hatası."""


class CLIPLoadError(FrameProcessorError):
    """CLIP modeli yüklenemedi."""


class VideoReadError(FrameProcessorError):
    """OpenCV video açma / okuma hatası."""


# ── Lazy-load model önbelleği ─────────────────────────────────────────────────

_clip_model     = None   # open_clip model
_clip_preprocess = None  # görüntü ön işleme transform
_clip_tokenizer  = None  # metin tokenizer
_text_cache: dict[str, torch.Tensor] = {}  # prompt → normalize edilmiş embedding


def _get_clip() -> tuple:
    """CLIP modelini döner; gerekiyorsa yükler."""
    global _clip_model, _clip_preprocess, _clip_tokenizer

    if _clip_model is None:
        device = _resolve_device(config.CLIP_DEVICE)
        logger.info(
            "CLIP modeli yükleniyor: '%s' (cihaz: %s) …",
            config.CLIP_MODEL_NAME,
            device,
        )
        try:
            # open_clip model adı "openai/clip-vit-base-patch32" →
            # model_name="ViT-B-32", pretrained="openai"
            model_name, pretrained = _parse_clip_model_name(config.CLIP_MODEL_NAME)
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device,
            )
            _clip_model.eval()
            _clip_tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise CLIPLoadError(f"CLIP modeli yüklenemedi: {exc}") from exc

        logger.info("CLIP modeli yüklendi.")

    return _clip_model, _clip_preprocess, _clip_tokenizer


def _parse_clip_model_name(full_name: str) -> tuple[str, str]:
    """
    "openai/clip-vit-base-patch32" → ("ViT-B-32", "openai")
    Formatı tanınmıyorsa orijinal değerlerle devam eder.
    """
    _known: dict[str, tuple[str, str]] = {
        "openai/clip-vit-base-patch32":  ("ViT-B-32",  "openai"),
        "openai/clip-vit-large-patch14": ("ViT-L-14",  "openai"),
        "laion/clip-vit-bigG-14-laion2b": ("ViT-bigG-14", "laion2b_s39b_b160k"),
    }
    if full_name in _known:
        return _known[full_name]
    # Serbest format: "org/model-name" → ("model-name", "org")
    parts = full_name.split("/", 1)
    if len(parts) == 2:
        return parts[1], parts[0]
    return full_name, "openai"


def _resolve_device(preferred: str) -> str:
    if preferred == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA bulunamadı; CLIP CPU modunda çalışacak.")
        return "cpu"
    return preferred


# ── Embedding hesaplama ───────────────────────────────────────────────────────

@torch.inference_mode()
def _image_embedding(image: Image.Image) -> torch.Tensor:
    """PIL Image → normalize edilmiş CLIP embedding (1-D tensor, CPU)."""
    model, preprocess, _ = _get_clip()
    device = next(model.parameters()).device
    tensor = preprocess(image).unsqueeze(0).to(device)
    emb = model.encode_image(tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu()


@torch.inference_mode()
def _text_embedding(prompt: str) -> torch.Tensor:
    """Metin prompt → normalize edilmiş CLIP embedding; process boyunca önbelleğe alınır."""
    if prompt in _text_cache:
        return _text_cache[prompt]
    model, _, tokenizer = _get_clip()
    device = next(model.parameters()).device
    tokens = tokenizer([prompt]).to(device)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    result = emb.squeeze(0).cpu()
    _text_cache[prompt] = result
    return result


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """İki normalize edilmiş vektör arasındaki kosinüs benzerliğini döner."""
    return float(torch.dot(a, b).item())


def _max_similarity(image_emb: torch.Tensor, prompts: list[str]) -> tuple[float, str]:
    """
    Bir görüntü embedding'inin verilen prompt listesiyle en yüksek benzerliğini bulur.
    Returns (max_score, best_prompt)
    """
    best_score  = -1.0
    best_prompt = ""
    for prompt in prompts:
        score = _cosine(image_emb, _text_embedding(prompt))
        if score > best_score:
            best_score  = score
            best_prompt = prompt
    return best_score, best_prompt


# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def process(video_input: VideoInput) -> list[FrameResult]:
    """
    VideoInput → List[FrameResult]

    Video dosyasını her FRAME_SAMPLE_INTERVAL saniyede örnekler,
    üç süzgeçten geçirir ve geçen frame'leri PNG olarak kaydeder.

    Parameters
    ----------
    video_input:
        input_handler.py tarafından üretilen standart girdi nesnesi.

    Returns
    -------
    Kronolojik sırayla geçen frame'lerin listesi.
    """
    video_path = Path(video_input.video_path)
    if not video_path.exists():
        raise VideoReadError(f"Video dosyası bulunamadı: {video_path}")

    clip_start = getattr(video_input, "clip_start", 0.0) or 0.0
    clip_end   = getattr(video_input, "clip_end",   None)

    logger.info(
        "Frame işleme başlıyor: '%s' (%.0f sn, ~%.0f frame örneklenecek%s)",
        video_input.title,
        video_input.duration,
        video_input.duration / config.FRAME_SAMPLE_INTERVAL,
        f", klip: {clip_start:.0f}s–{clip_end:.0f}s" if clip_start or clip_end else "",
    )

    # Prompt embedding'lerini ön-ısıt; model burada yüklenir
    _warmup_text_embeddings()

    cap = _open_video(video_path)
    results: list[FrameResult] = []

    stats = {
        "total_sampled":  0,
        "skipped_similar": 0,
        "skipped_empty":   0,
        "skipped_content": 0,
        "passed":          0,
    }

    try:
        results = _process_frames(cap, video_path, stats, clip_start, clip_end)
    finally:
        cap.release()

    logger.info(
        "Frame işleme tamamlandı — örneklenen: %d | "
        "atılan(benzer): %d | atılan(boş): %d | atılan(içeriksiz): %d | geçen: %d",
        stats["total_sampled"],
        stats["skipped_similar"],
        stats["skipped_empty"],
        stats["skipped_content"],
        stats["passed"],
    )

    return results


# ── Frame döngüsü ─────────────────────────────────────────────────────────────

def _process_frames(
    cap: cv2.VideoCapture,
    video_path: Path,
    stats: dict,
    clip_start: float = 0.0,
    clip_end: float | None = None,
) -> list[FrameResult]:
    """
    Frame okuma + süzgeç + kaydetme döngüsü.
    stats dict'i in-place günceller.
    clip_start/clip_end orijinal videodaki sınırlardır.
    Timestamp'ler clip_start'a göre sıfırlanır (0-tabanlı).
    """
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval_frames = max(1, int(round(fps * config.FRAME_SAMPLE_INTERVAL)))

    start_frame = int(clip_start * fps) if clip_start else 0
    end_frame   = int(clip_end   * fps) if clip_end   is not None else None

    prev_emb: torch.Tensor | None = None
    results: list[FrameResult]    = []
    frame_idx = start_frame

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        # Sadece örnekleme aralığındaki frame'leri oku; araları atla
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            break

        # Timestamp'i clip_start'a göre sıfırla (ses ile senkron kalır)
        timestamp = (frame_idx / fps) - clip_start
        frame_idx += interval_frames
        stats["total_sampled"] += 1

        image = _bgr_to_pil(bgr)
        img_emb = _image_embedding(image)

        # ── Süzgeç 1: CLIP benzerlik kontrolü ────────────────────────────────
        if prev_emb is not None:
            similarity = _cosine(img_emb, prev_emb)
            if similarity >= config.CLIP_SIMILARITY_THRESH:
                stats["skipped_similar"] += 1
                logger.debug(
                    "[%.2fs] Atıldı (benzer) — benzerlik=%.3f", timestamp, similarity
                )
                continue
        else:
            similarity = 0.0  # ilk frame; karşılaştırma yapılamaz

        prev_emb = img_emb  # bir sonraki frame için referans güncellenir

        # ── Süzgeç 2: Boş/avatar ekran kontrolü ──────────────────────────────
        empty_score, empty_prompt = _max_similarity(img_emb, _EMPTY_PROMPTS)
        content_score, content_prompt = _max_similarity(img_emb, _CONTENT_PROMPTS)

        # Boş skoru içerik skoru üzerindeyse anlamsız ekran
        if empty_score > content_score:
            stats["skipped_empty"] += 1
            logger.debug(
                "[%.2fs] Atıldı (boş/avatar) — empty=%.3f ('%s')",
                timestamp, empty_score, empty_prompt,
            )
            continue

        # ── Süzgeç 3: Bilgi içeriği kontrolü ─────────────────────────────────
        if content_score < config.CLIP_CONTENT_THRESH:
            stats["skipped_content"] += 1
            logger.debug(
                "[%.2fs] Atıldı (içeriksiz) — content_max=%.3f ('%s')",
                timestamp, content_score, content_prompt,
            )
            continue

        # ── Süzgeçleri geçti: kaydet ──────────────────────────────────────────
        frame_path = _save_frame(bgr, video_path.stem, timestamp)
        stats["passed"] += 1

        logger.debug(
            "[%.2fs] Geçti — benzerlik=%.3f | empty=%.3f | content=%.3f ('%s')",
            timestamp, similarity, empty_score, content_score, content_prompt,
        )

        results.append(
            FrameResult(
                timestamp=round(timestamp, 3),
                frame_path=str(frame_path),
                clip_scores={
                    "similarity":    round(similarity, 4),
                    "empty_max":     round(empty_score, 4),
                    "empty_prompt":  empty_prompt,
                    "content_max":   round(content_score, 4),
                    "content_prompt": content_prompt,
                },
            )
        )

    return results


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _open_video(video_path: Path) -> cv2.VideoCapture:
    """OpenCV ile video açar; başarısızsa VideoReadError fırlatır."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoReadError(
            f"Video açılamadı: {video_path}. "
            "Dosya bozuk veya codec desteklenmiyor olabilir."
        )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    logger.debug(
        "Video açıldı — %d frame, %.2f fps, ~%.0f sn",
        total_frames, fps, total_frames / fps if fps else 0,
    )
    return cap


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR array → PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _save_frame(bgr: np.ndarray, video_stem: str, timestamp: float) -> Path:
    """
    Frame'i PNG olarak workdir/ altına kaydeder.
    Dosya adı: {video_stem}_frame_{timestamp_ms}.png
    Örnek: toplanti_frame_00125400.png  (125.4 sn → 125400 ms)
    """
    ms      = int(timestamp * 1000)
    fname   = f"{video_stem}_frame_{ms:010d}.png"
    fpath   = config.WORK_DIR / fname
    success = cv2.imwrite(str(fpath), bgr)
    if not success:
        raise FrameProcessorError(f"Frame kaydedilemedi: {fpath}")
    return fpath


def _warmup_text_embeddings() -> None:
    """
    Tüm süzgeç prompt'larının embedding'lerini önceden hesaplar.
    Model ilk kez burada yüklenir; döngü içinde gecikme olmaz.
    """
    all_prompts = _EMPTY_PROMPTS + _CONTENT_PROMPTS
    logger.info(
        "CLIP metin embedding'leri ön-ısıtılıyor (%d prompt) …", len(all_prompts)
    )
    for prompt in all_prompts:
        _text_embedding(prompt)
    logger.debug("Metin embedding'leri hazır.")


# ── Model temizleme ───────────────────────────────────────────────────────────

def unload_models() -> None:
    """
    CLIP modelini ve metin önbelleğini bellekten siler.
    PARALLEL_AUDIO_VIDEO=False iken audio pipeline'dan önce/sonra çağrılabilir.
    """
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        del _clip_model
        del _clip_preprocess
        del _clip_tokenizer
        _clip_model = _clip_preprocess = _clip_tokenizer = None
        _text_cache.clear()
        logger.info("CLIP modeli ve metin önbelleği bellekten kaldırıldı.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache temizlendi.")
