"""
server.py
─────────
FastAPI uygulaması.

Endpoint'ler:
  POST /api/process/file        → video dosyası yükle, pipeline başlat
  POST /api/process/youtube     → YouTube URL'si ile pipeline başlat
  GET  /api/status/{job_id}     → iş durumu
  GET  /api/meetings            → arşiv listesi
  GET  /api/meetings/{id}       → tek toplantı
  GET  /api/meetings/{id}/download → PDF indir
  DELETE /api/meetings/{id}     → sil
  GET  /                        → index.html
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import time as _time
import traceback
import uuid
from pathlib import Path
from typing import Literal

# Her adımın tamamlandığında ulaşılan yüzde (GPU H100/H200 için ağırlıklar)
# Ses (Whisper+pyannote) en ağır, VLM frame sayısına göre değişir
_STEP_PCT = {
    "start":  2,
    "audio":  5,   # çalışıyor
    "frames": 40,  # audio bitti
    "vlm":    52,  # frames bitti
    "merge":  80,  # vlm bitti
    "notes":  82,  # merge bitti
    "pdf":    97,  # notes bitti
    "db":     99,  # pdf bitti
    "done":   100,
}

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from storage.database import (
    init_db, save_meeting, get_meeting, list_meetings, search_meetings, delete_meeting,
    save_job, update_job, get_job_status,
)
from pipeline import (
    input_handler,
    audio_processor,
    frame_processor,
    vlm_processor,
    merger,
    note_generator,
)
from output import pdf_generator

# ── Logger ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ── Uygulama ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Meeting Notes", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()
    logger.info("Uygulama başlatıldı. http://%s:%d", config.API_HOST, config.API_PORT)


# ── In-memory job takibi ──────────────────────────────────────────────────────

JobStatus = Literal["processing", "done", "error", "cancelled"]

class JobCancelledError(Exception):
    pass

class Job:
    def __init__(self):
        self.status:          JobStatus  = "processing"
        self.progress:        str        = "Başlatılıyor…"
        self.meeting_id:      int | None = None
        self.error:           str | None = None
        self.cancelled:       bool       = False
        self.percent:         int        = 0
        self.eta_seconds:     int | None = None
        self._started_at:     float      = _time.monotonic()
        self._video_duration: float      = 0.0

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelledError("İş kullanıcı tarafından iptal edildi.")

    def set_step(self, step: str, message: str) -> None:
        self.progress = message
        self.percent  = _STEP_PCT.get(step, self.percent)
        elapsed = _time.monotonic() - self._started_at
        pct = self.percent
        if pct > 5 and elapsed > 0:
            self.eta_seconds = int(elapsed / pct * (100 - pct))
        elif self._video_duration > 0:
            # Henüz yeterli veri yok — video süresine göre kaba tahmin
            # H100/H200 için yaklaşık: ses*0.1 + VLM dominant + sabit overhead
            est = self._video_duration * 1.3 + 90
            self.eta_seconds = int(est)

jobs: dict[str, Job] = {}


def _new_job() -> tuple[str, Job]:
    job_id  = uuid.uuid4().hex
    job     = Job()
    jobs[job_id] = job
    save_job(job_id)
    return job_id, job


# ── Pydantic modelleri ────────────────────────────────────────────────────────

class YoutubeRequest(BaseModel):
    url: str
    start_time: float | None = None   # saniye; None → videonun başından
    end_time:   float | None = None   # saniye; None → videonun sonuna kadar


class StatusResponse(BaseModel):
    status:      str
    progress:    str
    meeting_id:  int | None = None
    error:       str | None = None
    percent:     int        = 0
    eta_seconds: int | None = None


# ── Pipeline çalıştırıcı ──────────────────────────────────────────────────────

def _run_pipeline(job: Job, job_id: str, video_input, source_type: str, source_url: str | None = None) -> None:
    """
    Tüm pipeline adımlarını sırayla çalıştırır.
    Her adımda job.progress güncellenir.
    Herhangi bir hata job.status = "error" yapar ve durur.
    İptal isteği gelirse adımlar arasında JobCancelledError fırlatılır.
    """
    job._video_duration = video_input.duration
    job.set_step("start", "Başlatılıyor…")

    try:
        if config.PARALLEL_AUDIO_VIDEO:
            # ── Paralel mod: ses + görüntü eş zamanlı ────────────────────────
            job.set_step("audio", "Ses + görüntü paralel işleniyor…")
            logger.info("[%s] Paralel pipeline başlıyor.", id(job))

            _audio_exc: Exception | None = None
            _video_exc: Exception | None = None

            def _audio_task():
                return audio_processor.process(video_input)

            def _video_task():
                frames = frame_processor.process(video_input)
                frame_processor.unload_models()
                vlm = vlm_processor.process(frames)
                vlm_processor.unload_models()
                return vlm

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                audio_fut = ex.submit(_audio_task)
                video_fut = ex.submit(_video_task)
                job.check_cancelled()
                annotated_segments = audio_fut.result()
                job.set_step("vlm", "Görüntü analizi tamamlanıyor…")
                vlm_results = video_fut.result()

            audio_processor.unload_models()
            logger.info("[%s] Paralel pipeline tamamlandı.", id(job))
        else:
            # ── Sıralı mod ───────────────────────────────────────────────────
            job.check_cancelled()
            job.set_step("audio", "Ses transkripti ve konuşmacı analizi yapılıyor…")
            logger.info("[%s] %s", id(job), job.progress)
            annotated_segments = audio_processor.process(video_input)
            audio_processor.unload_models()

            job.check_cancelled()
            job.set_step("frames", "Görüntüler analiz ediliyor…")
            logger.info("[%s] %s", id(job), job.progress)
            frame_results = frame_processor.process(video_input)
            frame_processor.unload_models()

            job.check_cancelled()
            job.set_step("vlm", "Ekranlar yorumlanıyor (VLM)…")
            logger.info("[%s] %s", id(job), job.progress)
            vlm_results = vlm_processor.process(frame_results)
            vlm_processor.unload_models()

        # ── 4. Birleştirme ────────────────────────────────────────────────────
        job.check_cancelled()
        job.set_step("merge", "Ses ve görüntü birleştiriliyor…")
        logger.info("[%s] %s", id(job), job.progress)
        merged_items = merger.merge(annotated_segments, vlm_results)

        # ── 5. Not üretimi ────────────────────────────────────────────────────
        job.check_cancelled()
        job.set_step("notes", "Notlar üretiliyor (Mistral)…")
        logger.info("[%s] %s", id(job), job.progress)
        notes = note_generator.generate(
            items=merged_items,
            title=video_input.title,
            duration=video_input.duration,
        )
        note_generator.unload_models()

        # ── 6. PDF ────────────────────────────────────────────────────────────
        job.check_cancelled()
        job.set_step("pdf", "PDF oluşturuluyor…")
        logger.info("[%s] %s", id(job), job.progress)
        pdf_filename = f"{uuid.uuid4().hex[:8]}_{_safe_stem(video_input.title)}.pdf"
        pdf_path     = str(config.OUTPUT_DIR / pdf_filename)
        pdf_generator.generate(notes, pdf_path)

        # ── 7. Veritabanı ─────────────────────────────────────────────────────
        job.set_step("db", "Arşive kaydediliyor…")
        meeting_id = save_meeting(
            notes=notes,
            pdf_path=pdf_path,
            source_type=source_type,
            source_url=source_url,
        )

        job.meeting_id = meeting_id
        job.status     = "done"
        job.set_step("done", "Tamamlandı")
        job.eta_seconds = 0
        update_job(job_id, "done", "Tamamlandı", meeting_id=meeting_id)
        logger.info("[%s] Pipeline tamamlandı — meeting_id=%d", id(job), meeting_id)

    except JobCancelledError:
        job.status   = "cancelled"
        job.progress = "İptal edildi"
        update_job(job_id, "cancelled", "İptal edildi")
        logger.info("[%s] İş iptal edildi.", id(job))

    except Exception as exc:
        job.status   = "error"
        job.progress = "Hata oluştu"
        job.error    = str(exc)
        update_job(job_id, "error", "Hata oluştu", error=str(exc))
        logger.error(
            "[%s] Pipeline hatası: %s\n%s",
            id(job),
            exc,
            traceback.format_exc(),
        )


# ── Endpoint'ler ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Ana sayfa — index.html döner."""
    html_path = config.STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html bulunamadı.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/process/file")
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start_time: float | None = Form(None),
    end_time:   float | None = Form(None),
):
    """
    Video dosyasını yükler ve pipeline'ı arka planda başlatır.
    """
    # Dosya boyutu kontrolü (stream bitmeden kontrol edilemez;
    # uploads/ dizinine kaydederken boyutu ölçeceğiz)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Desteklenmeyen format '{ext}'. "
                f"Kabul edilen: {', '.join(sorted(config.ALLOWED_VIDEO_EXTENSIONS))}"
            ),
        )

    # Geçici konuma kaydet
    tmp_path = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    try:
        with tmp_path.open("wb") as fout:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunk
                fout.write(chunk)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Dosya kaydedilemedi: {exc}")

    size_mb = tmp_path.stat().st_size / (1024 ** 2)
    if size_mb > config.MAX_UPLOAD_MB:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"Dosya çok büyük: {size_mb:.1f} MB (sınır: {config.MAX_UPLOAD_MB} MB).",
        )

    logger.info("Dosya yüklendi: %s (%.1f MB)", file.filename, size_mb)

    job_id, job = _new_job()

    def _task():
        try:
            video_input = input_handler.handle_file(
                tmp_path,
                start_time=start_time,
                end_time=end_time,
            )
            _run_pipeline(job, job_id, video_input, source_type="file")
        except Exception as exc:
            job.status = "error"
            job.error  = str(exc)
            update_job(job_id, "error", "Hata oluştu", error=str(exc))
            logger.error("Dosya pipeline başlatılamadı: %s", exc)

    background_tasks.add_task(_task)
    logger.info(
        "Dosya işi başlatıldı: job_id=%s%s",
        job_id,
        f" (klip: {start_time}s – {end_time}s)" if start_time is not None or end_time is not None else "",
    )
    return {"job_id": job_id}


@app.post("/api/process/youtube")
def process_youtube(
    body: YoutubeRequest,
    background_tasks: BackgroundTasks,
):
    """
    YouTube URL'si için pipeline'ı arka planda başlatır.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="URL boş olamaz.")

    job_id, job = _new_job()

    def _task():
        try:
            video_input = input_handler.handle_youtube(
                url,
                start_time=body.start_time,
                end_time=body.end_time,
            )
            _run_pipeline(job, job_id, video_input, source_type="youtube", source_url=url)
        except Exception as exc:
            job.status = "error"
            job.error  = str(exc)
            update_job(job_id, "error", "Hata oluştu", error=str(exc))
            logger.error("YouTube pipeline başlatılamadı: %s", exc)

    background_tasks.add_task(_task)
    logger.info(
        "YouTube işi başlatıldı: job_id=%s, url=%s%s",
        job_id,
        url,
        f" (klip: {body.start_time}s – {body.end_time}s)"
        if body.start_time is not None or body.end_time is not None else "",
    )
    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Devam eden işi iptal eder."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job bulunamadı: {job_id}")
    if job.status != "processing":
        raise HTTPException(status_code=409, detail=f"İş zaten tamamlanmış: {job.status}")
    job.cancelled = True
    logger.info("İptal isteği alındı: job_id=%s", job_id)
    return {"cancelled": True, "job_id": job_id}


@app.get("/api/status/{job_id}", response_model=StatusResponse)
def job_status(job_id: str):
    """İş durumunu döner. Bellekte yoksa DB'den alır."""
    job = jobs.get(job_id)
    if job is not None:
        return StatusResponse(
            status=job.status,
            progress=job.progress,
            meeting_id=job.meeting_id,
            error=job.error,
            percent=job.percent,
            eta_seconds=job.eta_seconds,
        )
    # Sunucu yeniden başlatılmış olabilir — DB'den kontrol et
    db_job = get_job_status(job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail=f"Job bulunamadı: {job_id}")
    return StatusResponse(
        status=db_job["status"],
        progress=db_job["progress"],
        meeting_id=db_job.get("meeting_id"),
        error=db_job.get("error"),
        percent=100 if db_job["status"] == "done" else 0,
    )


@app.get("/api/meetings")
def api_list_meetings(limit: int = 50, q: str = ""):
    """Arşiv listesi; q parametresi varsa arama yapar."""
    if q.strip():
        return search_meetings(q.strip())
    return list_meetings(limit=limit)


@app.get("/api/meetings/{meeting_id}")
def api_get_meeting(meeting_id: int):
    """Tek toplantı kaydı."""
    meeting = get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail=f"Toplantı bulunamadı: {meeting_id}")
    return meeting


@app.get("/api/meetings/{meeting_id}/download")
def api_download_pdf(meeting_id: int):
    """PDF dosyasını indir."""
    meeting = get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail=f"Toplantı bulunamadı: {meeting_id}")

    pdf_path = Path(meeting["pdf_path"])
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF dosyası sunucuda bulunamadı. Silinmiş olabilir.",
        )

    safe_name = _safe_stem(meeting["title"]) + ".pdf"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=safe_name,
    )


@app.delete("/api/meetings/{meeting_id}")
def api_delete_meeting(meeting_id: int):
    """Toplantı kaydını sil."""
    deleted = delete_meeting(meeting_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Toplantı bulunamadı: {meeting_id}")
    return {"deleted": True, "meeting_id": meeting_id}


# ── Küçük yardımcılar ─────────────────────────────────────────────────────────

def _safe_stem(name: str) -> str:
    """Dosya adı için geçersiz karakterleri kaldırır."""
    import re
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name[:60].strip("_") or "toplanti"
