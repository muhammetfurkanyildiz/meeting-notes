"""
pdf_generator.py
────────────────
MeetingNotes nesnesini alır, Jinja2 ile HTML şablonu render eder,
WeasyPrint ile PDF'e dönüştürür.

Çıktı PDF yapısı:
  1. Kapak  — başlık / tarih / süre / katılımcılar
  2. Özet
  3. Alınan Kararlar  (varsa)
  4. Action Items     (varsa)
  5. Kronolojik Akış  — speech + frame sıralı
  6. Teknik Anlar     (varsa)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from jinja2 import Environment, BaseLoader
from weasyprint import HTML, CSS

import config
from pipeline.note_generator import MeetingNotes
from pipeline.merger import MergedItem

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("pdf_generator")

# ── Hata sınıfları ────────────────────────────────────────────────────────────

class PDFGeneratorError(Exception):
    """PDF üretim hatası."""

# ── Ana giriş noktası ─────────────────────────────────────────────────────────

def generate(notes: MeetingNotes, output_path: str) -> str:
    """
    MeetingNotes → PDF dosyası.

    Parameters
    ----------
    notes:
        note_generator.generate() çıktısı.
    output_path:
        Oluşturulacak PDF'in tam yolu.

    Returns
    -------
    Kaydedilen PDF dosyasının mutlak yolu (str).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("PDF üretimi başlıyor: '%s'", notes.title)

    html_content = _render_html(notes)

    try:
        css_sources = [CSS(string=_INLINE_CSS)]
        if config.PDF_CSS_FILE.exists():
            css_sources.append(CSS(filename=str(config.PDF_CSS_FILE)))
            logger.debug("Özel CSS yüklendi: %s", config.PDF_CSS_FILE)

        HTML(string=html_content, base_url=str(config.WORK_DIR)).write_pdf(
            target=str(out),
            stylesheets=css_sources,
        )
    except Exception as exc:
        raise PDFGeneratorError(f"PDF yazılamadı: {exc}") from exc

    size_kb = out.stat().st_size / 1024
    logger.info("PDF oluşturuldu: %s (%.1f KB)", out.name, size_kb)
    return str(out.resolve())


# ── HTML render ───────────────────────────────────────────────────────────────

_CONTENT_TYPE_LABELS: dict[str, str] = {
    "terminal": "Terminal / Komut Satırı",
    "code":     "Kod",
    "slide":    "Sunum Slaytı",
    "diagram":  "Diyagram / Grafik",
    "ui":       "Uygulama Ekranı",
    "unknown":  "Ekran Görüntüsü",
}
_CONTENT_TYPE_ICONS: dict[str, str] = {
    "terminal": "💻",
    "code":     "🖥️",
    "slide":    "📊",
    "diagram":  "📈",
    "ui":       "🖱️",
}
_DEFAULT_ICON = "📸"


def _render_html(notes: MeetingNotes) -> str:
    env      = Environment(loader=BaseLoader(), autoescape=True)
    env.filters["fmt_ts"]       = _fmt_ts
    env.filters["b64_image"]    = _b64_image
    env.filters["content_label"] = lambda ct: _CONTENT_TYPE_LABELS.get(ct or "", "Ekran Görüntüsü")
    env.filters["content_icon"]  = lambda ct: _CONTENT_TYPE_ICONS.get(ct or "", _DEFAULT_ICON)

    template = env.from_string(_HTML_TEMPLATE)
    return template.render(notes=notes)


def _fmt_ts(seconds: float | None) -> str:
    """123.4 → '02:03'"""
    if seconds is None:
        return "00:00"
    total = int(seconds)
    m, s  = divmod(total, 60)
    h, m  = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _b64_image(path: str | None) -> str | None:
    """Görüntü dosyasını base64 data URI'ye dönüştürür; dosya yoksa None döner."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{data}"
    except Exception:
        return None


# ── CSS ───────────────────────────────────────────────────────────────────────

_INLINE_CSS = """
/* ── Temel ── */
@page {
    size: A4;
    margin: 2cm 2.2cm 2.5cm 2.2cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #999;
        font-family: 'Inter', Arial, sans-serif;
    }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a1a;
    background: #fff;
}

/* ── Kapak ── */
.cover {
    page-break-after: always;
    padding-top: 6cm;
    text-align: center;
}
.cover-title {
    font-size: 26pt;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.5cm;
    line-height: 1.25;
}
.cover-meta {
    font-size: 11pt;
    color: #555;
    margin-bottom: 0.3cm;
}
.cover-divider {
    width: 6cm;
    height: 3px;
    background: #4f46e5;
    margin: 1cm auto;
    border-radius: 2px;
}
.cover-participants-title {
    font-size: 10pt;
    font-weight: 600;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3cm;
}
.cover-participants {
    font-size: 11pt;
    color: #333;
}

/* ── Bölüm başlıkları ── */
h2 {
    font-size: 15pt;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 2px solid #e0e0f0;
    padding-bottom: 0.2cm;
    margin-top: 0.8cm;
    margin-bottom: 0.5cm;
    page-break-after: avoid;
}
h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #333;
    margin-bottom: 0.25cm;
    page-break-after: avoid;
}

/* ── Özet ── */
.summary-text {
    background: #f5f5ff;
    border-left: 4px solid #4f46e5;
    padding: 0.4cm 0.6cm;
    border-radius: 0 6px 6px 0;
    color: #222;
}

/* ── Kararlar ── */
.decisions-list {
    list-style: none;
    padding: 0;
}
.decisions-list li {
    padding: 0.25cm 0 0.25cm 1.2em;
    border-bottom: 1px solid #f0f0f0;
    position: relative;
}
.decisions-list li::before {
    content: "✓";
    position: absolute;
    left: 0;
    color: #4f46e5;
    font-weight: 700;
}

/* ── Action Items tablosu ── */
.action-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9.5pt;
    margin-top: 0.3cm;
}
.action-table th {
    background: #4f46e5;
    color: #fff;
    padding: 0.2cm 0.35cm;
    text-align: left;
    font-weight: 600;
}
.action-table td {
    padding: 0.2cm 0.35cm;
    border-bottom: 1px solid #e8e8f0;
    vertical-align: top;
}
.action-table tr:nth-child(even) td {
    background: #f8f8ff;
}
.due-badge {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border-radius: 4px;
    padding: 0 0.2em;
    font-size: 9pt;
}

/* ── Kronolojik akış ── */
.timeline { margin-top: 0.4cm; }

.timeline-item {
    display: flex;
    gap: 0.4cm;
    margin-bottom: 0.45cm;
    page-break-inside: avoid;
}
.ts-badge {
    flex-shrink: 0;
    width: 1.3cm;
    font-size: 8.5pt;
    color: #888;
    padding-top: 0.05cm;
    font-variant-numeric: tabular-nums;
    font-family: 'Courier New', monospace;
}
.speech-body {
    flex: 1;
}
.speaker-name {
    font-weight: 700;
    color: #1a1a2e;
    font-size: 9.5pt;
    margin-right: 0.2em;
}
.speech-text {
    color: #333;
}

/* Frame item */
.frame-body {
    flex: 1;
    background: #f9f9fb;
    border: 1px solid #e0e0ea;
    border-radius: 6px;
    padding: 0.3cm 0.4cm;
}
.frame-header {
    font-size: 9pt;
    font-weight: 600;
    color: #4f46e5;
    margin-bottom: 0.2cm;
}
.frame-image {
    max-width: 100%;
    max-height: 7cm;
    border-radius: 4px;
    border: 1px solid #ddd;
    display: block;
    margin-bottom: 0.2cm;
}
.frame-description {
    font-size: 9.5pt;
    color: #444;
    margin-top: 0.15cm;
}
.code-block {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    background: #1e1e2e;
    color: #cdd6f4;
    padding: 0.35cm 0.45cm;
    border-radius: 5px;
    white-space: pre-wrap;
    word-break: break-all;
    margin-top: 0.2cm;
    line-height: 1.5;
}

/* ── Teknik Anlar ── */
.tech-moment {
    display: flex;
    gap: 0.4cm;
    margin-bottom: 0.6cm;
    page-break-inside: avoid;
    border: 1px solid #e0e0ea;
    border-radius: 6px;
    padding: 0.35cm 0.45cm;
    background: #fafafa;
}
.tech-moment-image {
    flex-shrink: 0;
    width: 5cm;
}
.tech-moment-image img {
    max-width: 100%;
    border-radius: 4px;
    border: 1px solid #ddd;
}
.tech-moment-body { flex: 1; }
.tech-ts {
    font-size: 8.5pt;
    font-family: 'Courier New', monospace;
    color: #888;
    margin-bottom: 0.1cm;
}
.tech-desc { font-size: 10pt; color: #222; }

/* ── Fallback kutusu ── */
.raw-fallback {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt;
    background: #fff8e1;
    border-left: 3px solid #f59e0b;
    padding: 0.3cm 0.4cm;
    white-space: pre-wrap;
    border-radius: 0 4px 4px 0;
    margin-bottom: 0.3cm;
}

/* ── Yardımcı sınıflar ── */
.section { margin-bottom: 0.8cm; }
.no-content {
    font-size: 9.5pt;
    color: #999;
    font-style: italic;
}
"""

# ── HTML şablonu ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>{{ notes.title }}</title>
</head>
<body>

{# ════════════════════════════════════════════════════════════════
   1. KAPAK
   ════════════════════════════════════════════════════════════════ #}
<div class="cover">
  <div class="cover-title">{{ notes.title }}</div>
  <div class="cover-meta">📅 {{ notes.date }}</div>
  <div class="cover-meta">⏱ {{ notes.duration_str }}</div>
  <div class="cover-divider"></div>
  {% if notes.speakers %}
  <div class="cover-participants-title">Katılımcılar</div>
  <div class="cover-participants">{{ notes.speakers | join("  ·  ") }}</div>
  {% endif %}
</div>

{# ════════════════════════════════════════════════════════════════
   2. GENEL ÖZET
   ════════════════════════════════════════════════════════════════ #}
<div class="section">
  <h2>📋 Genel Özet</h2>
  <div class="summary-text">{{ notes.summary }}</div>
</div>

{# ════════════════════════════════════════════════════════════════
   3. ALINAN KARARLAR
   ════════════════════════════════════════════════════════════════ #}
{% if notes.decisions %}
<div class="section">
  <h2>✅ Alınan Kararlar</h2>
  <ul class="decisions-list">
    {% for d in notes.decisions %}
    <li>{{ d }}</li>
    {% endfor %}
  </ul>
</div>
{% endif %}

{# ════════════════════════════════════════════════════════════════
   4. ACTION ITEMS
   ════════════════════════════════════════════════════════════════ #}
{% if notes.action_items %}
<div class="section">
  <h2>🎯 Action Items</h2>
  <table class="action-table">
    <thead>
      <tr>
        <th style="width:22%">Kim</th>
        <th>Ne yapacak</th>
        <th style="width:18%">Ne zaman</th>
      </tr>
    </thead>
    <tbody>
      {% for ai in notes.action_items %}
      <tr>
        <td><strong>{{ ai.owner }}</strong></td>
        <td>{{ ai.task }}</td>
        <td>
          {% if ai.due %}
          <span class="due-badge">{{ ai.due }}</span>
          {% else %}
          <span style="color:#bbb">—</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}

{# ════════════════════════════════════════════════════════════════
   5. KRONOLOJİK AKIŞ
   ════════════════════════════════════════════════════════════════ #}
<div class="section">
  <h2>🕐 Kronolojik Akış</h2>
  {% if notes.transcript %}
  <div class="timeline">
    {% for item in notes.transcript %}

    {# ── Speech item ── #}
    {% if item.item_type == "speech" %}
    <div class="timeline-item">
      <div class="ts-badge">{{ item.timestamp | fmt_ts }}</div>
      <div class="speech-body">
        <span class="speaker-name">{{ item.speaker }}:</span>
        <span class="speech-text">{{ item.text }}</span>
      </div>
    </div>

    {# ── Frame item ── #}
    {% elif item.item_type == "frame" %}
    {% set img_src = item.frame_path | b64_image %}
    <div class="timeline-item">
      <div class="ts-badge">{{ item.timestamp | fmt_ts }}</div>
      <div class="frame-body">
        <div class="frame-header">
          {{ item.content_type | content_icon }} {{ item.content_type | content_label }}
        </div>
        {% if img_src %}
        <img class="frame-image" src="{{ img_src }}" alt="Ekran görüntüsü {{ item.timestamp | fmt_ts }}">
        {% endif %}
        {% if item.description %}
        <div class="frame-description">{{ item.description }}</div>
        {% endif %}
        {% if item.extracted_text %}
        <div class="code-block">{{ item.extracted_text }}</div>
        {% endif %}
      </div>
    </div>
    {% endif %}

    {% endfor %}
  </div>
  {% else %}
  <p class="no-content">Transkript verisi bulunmuyor.</p>
  {% endif %}
</div>

{# ════════════════════════════════════════════════════════════════
   6. TEKNİK ANLAR
   ════════════════════════════════════════════════════════════════ #}
{% if notes.technical_moments %}
<div class="section">
  <h2>🔧 Teknik Anlar</h2>
  {% for tm in notes.technical_moments %}
  {% set img_src = tm.frame_path | b64_image %}
  <div class="tech-moment">
    {% if img_src %}
    <div class="tech-moment-image">
      <img src="{{ img_src }}" alt="Teknik an {{ tm.timestamp | fmt_ts }}">
    </div>
    {% endif %}
    <div class="tech-moment-body">
      <div class="tech-ts">⏱ {{ tm.timestamp | fmt_ts }}</div>
      <div class="tech-desc">{{ tm.description }}</div>
    </div>
  </div>
  {% endfor %}
</div>
{% endif %}

{# ════════════════════════════════════════════════════════════════
   Ham LLM çıktısı (JSON parse edilemeyenler)
   ════════════════════════════════════════════════════════════════ #}
{% if notes.raw_llm_fallback %}
<div class="section">
  <h2>📄 Ham Notlar</h2>
  <p style="font-size:9pt;color:#888;margin-bottom:0.3cm;">
    Aşağıdaki içerik yapılandırılamadı; ham model çıktısı olarak kaydedildi.
  </p>
  {% for raw in notes.raw_llm_fallback %}
  <div class="raw-fallback">{{ raw }}</div>
  {% endfor %}
</div>
{% endif %}

</body>
</html>
"""
