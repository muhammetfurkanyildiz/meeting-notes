"""
database.py
───────────
SQLite ile toplantı arşivi yönetimi.
Tüm erişim thread-safe connection factory üzerinden yapılır;
her çağrı kendi bağlantısını açıp kapatır.

Tablo:  meetings
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import config
from pipeline.note_generator import MeetingNotes

# ── Job şeması ────────────────────────────────────────────────────────────────

_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'processing',
    progress   TEXT NOT NULL DEFAULT 'Başlatılıyor…',
    meeting_id INTEGER,
    error      TEXT,
    created_at TEXT NOT NULL
);
"""

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("database")

# ── Şema ─────────────────────────────────────────────────────────────────────

_CREATE_MEETINGS = """
CREATE TABLE IF NOT EXISTS meetings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    duration    REAL    NOT NULL DEFAULT 0.0,
    source_type TEXT    NOT NULL,
    source_url  TEXT,
    pdf_path    TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    speakers    TEXT    NOT NULL DEFAULT '[]',
    summary     TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS meetings_fts
USING fts5(
    title,
    summary,
    content='meetings',
    content_rowid='id'
);
"""

# FTS'yi meetings tablosuyla senkron tut
_CREATE_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS meetings_ai AFTER INSERT ON meetings BEGIN
    INSERT INTO meetings_fts(rowid, title, summary)
    VALUES (new.id, new.title, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS meetings_ad AFTER DELETE ON meetings BEGIN
    INSERT INTO meetings_fts(meetings_fts, rowid, title, summary)
    VALUES ('delete', old.id, old.title, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS meetings_au AFTER UPDATE ON meetings BEGIN
    INSERT INTO meetings_fts(meetings_fts, rowid, title, summary)
    VALUES ('delete', old.id, old.title, old.summary);
    INSERT INTO meetings_fts(rowid, title, summary)
    VALUES (new.id, new.title, new.summary);
END;
"""

# ── Hata sınıfları ────────────────────────────────────────────────────────────

class DatabaseError(Exception):
    """Genel veritabanı hatası."""


# ── Bağlantı yönetimi ─────────────────────────────────────────────────────────

@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """
    Her çağrıda yeni bir bağlantı açar, işlem sonunda kapatır.
    check_same_thread=False: FastAPI worker thread'lerinden güvenli erişim.
    row_factory ile sonuçlar dict olarak döner.
    """
    conn = sqlite3.connect(
        str(config.DB_PATH),
        check_same_thread=False,
        timeout=10,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # eş zamanlı okuma/yazma için
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Başlatma ──────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Veritabanı tablolarını ve FTS indeksini oluşturur.
    Tablolar zaten varsa sessizce geçer.
    Uygulama başlangıcında (main.py / server.py startup) çağrılmalıdır.
    """
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        conn.execute(_CREATE_MEETINGS)
        conn.execute(_CREATE_FTS)
        conn.execute(_CREATE_JOBS)
        # Trigger'ları tek tek çalıştır (executescript transaction açar, biz istemiyoruz)
        for stmt in _CREATE_TRIGGERS.strip().split("\n\n"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

    logger.info("Veritabanı hazır: %s", config.DB_PATH)


# ── Yazma ─────────────────────────────────────────────────────────────────────

def save_meeting(
    notes: MeetingNotes,
    pdf_path: str,
    source_type: str,
    source_url: str | None = None,
) -> int:
    """
    MeetingNotes nesnesini veritabanına kaydeder.

    Parameters
    ----------
    notes:
        note_generator.generate() çıktısı.
    pdf_path:
        Üretilen PDF dosyasının mutlak yolu.
    source_type:
        "file" veya "youtube".
    source_url:
        YouTube linki (source_type=="youtube" ise); dosya için None.

    Returns
    -------
    Yeni kaydın id'si.
    """
    speakers_json = json.dumps(notes.speakers, ensure_ascii=False)
    created_at    = datetime.now().isoformat(timespec="seconds")

    # duration → float (MeetingNotes.duration_str değil, ham saniye gerekiyor;
    # duration_str "1d 30sn" gibi formatlanmış. Transcript'ten tahmin edelim.)
    duration_sec = _estimate_duration(notes)

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO meetings
                (title, date, duration, source_type, source_url, pdf_path,
                 created_at, speakers, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notes.title,
                notes.date,
                duration_sec,
                source_type,
                source_url,
                pdf_path,
                created_at,
                speakers_json,
                notes.summary,
            ),
        )
        meeting_id = cursor.lastrowid

    logger.info(
        "Toplantı kaydedildi — id: %d, başlık: '%s'", meeting_id, notes.title
    )
    return meeting_id


# ── Okuma ─────────────────────────────────────────────────────────────────────

def get_meeting(meeting_id: int) -> dict | None:
    """
    Tek bir toplantı kaydını döner.

    Returns
    -------
    dict veya None (kayıt bulunamazsa).
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()

    if row is None:
        logger.debug("Toplantı bulunamadı: id=%d", meeting_id)
        return None

    return _row_to_dict(row)


def list_meetings(limit: int = 50) -> list[dict]:
    """
    Toplantıları en yeniden eskiye sıralar ve döner.

    Parameters
    ----------
    limit:
        Maksimum döndürülecek kayıt sayısı (varsayılan 50).
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM meetings ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    result = [_row_to_dict(r) for r in rows]
    logger.debug("Toplantı listesi döndürüldü: %d kayıt.", len(result))
    return result


def search_meetings(query: str) -> list[dict]:
    """
    Başlık ve özet alanlarında tam metin araması yapar (FTS5).

    Parameters
    ----------
    query:
        Arama terimi. Boşsa tüm kayıtlar döner.

    Returns
    -------
    Eşleşen toplantılar, en yeniden eskiye sıralı.
    """
    query = query.strip()

    if not query:
        return list_meetings()

    # FTS5 özel karakterlerini temizle
    safe_query = _sanitize_fts_query(query)

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT m.*
            FROM   meetings m
            JOIN   meetings_fts f ON m.id = f.rowid
            WHERE  meetings_fts MATCH ?
            ORDER  BY m.created_at DESC
            """,
            (safe_query,),
        ).fetchall()

    result = [_row_to_dict(r) for r in rows]
    logger.debug(
        "Arama tamamlandı — sorgu: '%s', sonuç: %d kayıt.", query, len(result)
    )
    return result


# ── Silme ─────────────────────────────────────────────────────────────────────

def delete_meeting(meeting_id: int) -> bool:
    """
    Toplantı kaydını siler.

    Returns
    -------
    True → silindi, False → kayıt zaten yoktu.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM meetings WHERE id = ?", (meeting_id,)
        )
        deleted = cursor.rowcount > 0

    if deleted:
        logger.info("Toplantı silindi: id=%d", meeting_id)
    else:
        logger.warning("Silinecek toplantı bulunamadı: id=%d", meeting_id)

    return deleted


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def save_job(job_id: str) -> None:
    """Yeni iş kaydı oluşturur."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO jobs (job_id, created_at) VALUES (?, ?)",
            (job_id, datetime.now().isoformat(timespec="seconds")),
        )


def update_job(
    job_id: str,
    status: str,
    progress: str,
    meeting_id: int | None = None,
    error: str | None = None,
) -> None:
    """İş durumunu günceller."""
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, progress=?, meeting_id=?, error=? WHERE job_id=?",
            (status, progress, meeting_id, error, job_id),
        )


def get_job_status(job_id: str) -> dict | None:
    """İş durumunu döner; bulunamazsa None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict:
    """sqlite3.Row → dict; speakers JSON string'ini listeye çevirir."""
    d = dict(row)
    try:
        d["speakers"] = json.loads(d.get("speakers") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["speakers"] = []
    return d


def _estimate_duration(notes: MeetingNotes) -> float:
    """
    MeetingNotes'tan saniye cinsinden süreyi tahmin eder.
    Transcript'teki en son item'ın timestamp'ini kullanır.
    Transcript boşsa 0.0 döner.
    """
    if not notes.transcript:
        return 0.0
    last = notes.transcript[-1]
    end  = getattr(last, "end_timestamp", None) or last.timestamp
    return float(end)


def _sanitize_fts_query(query: str) -> str:
    """
    FTS5 sorgusundaki özel karakterleri temizler.
    Kullanıcı girişinin doğrudan MATCH'e gönderilmesi ayrıştırma hatası verebilir.
    Basit yaklaşım: tırnak dışındaki özel karakterleri boşlukla değiştir,
    her kelimeyi OR'la birleştir.
    """
    # Tırnaklı ifadeleri koru; geri kalanı kelime kelime işle
    tokens = query.split()
    safe_tokens: list[str] = []
    for token in tokens:
        # FTS5 özel: AND OR NOT ( ) * " ^
        cleaned = token.replace('"', "").replace("(", "").replace(")", "")
        cleaned = cleaned.replace("*", "").replace("^", "")
        if cleaned.strip():
            safe_tokens.append(f'"{cleaned}"')  # tırnak içine al → phrase match
    return " OR ".join(safe_tokens) if safe_tokens else '""'
