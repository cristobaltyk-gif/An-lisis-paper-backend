"""
reporte_pdf.py — Genera el PDF del informe de revisión crítica (modo "Subir archivo").

Uso desde main.py:
    from modules.reporte_pdf import build_review_pdf, review_pdf_filename
    pdf_bytes = build_review_pdf(review, lang="es")   # lang: "es" | "en"
    nombre    = review_pdf_filename(review, lang="es")

Recibe el mismo dict que produce modules.analysis (español) o su traducción
(translate_review, inglés): las claves son las mismas en ambos idiomas; lo que
cambia con 'lang' son los títulos de sección y las etiquetas del PDF.

Notas de diseño:
- Usa las fuentes estándar de PDF (Helvetica), que no dependen de archivos de
  fuente en el servidor. Como esas fuentes no tienen todos los símbolos, el texto
  pasa antes por _limpiar(): símbolos frecuentes en medicina (≥, ≤, β, →, ligaduras
  de PDF como "ﬁ") se convierten a su equivalente en texto y lo que no tenga
  equivalente se muestra como "?" en vez de un recuadro negro.
- No usa tablas para el contenido (solo para la etiqueta de recomendación), para
  que un texto largo nunca rompa el diseño de la página.
"""
import io
import re
import unicodedata
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

NAVY = colors.HexColor("#12284a")
GRAY = colors.HexColor("#5a6472")
LINK = "#1a56b0"

LABELS = {
    "es": {
        "doc": "Revisión crítica de manuscrito",
        "tipo": "Tipo de estudio",
        "recomendacion": "Recomendación",
        "resumen": "Datos y resumen del estudio",
        "diseno": "Diseño",
        "poblacion": "Población y datos",
        "desenlace": "Desenlace",
        "exposicion": "Exposición o intervención",
        "resultados": "Resultados centrales",
        "conclusion_autores": "Conclusión de los autores",
        "evaluacion": "Evaluación general",
        "fortalezas": "Fortalezas",
        "mayores": "Comentarios mayores",
        "menores": "Comentarios menores",
        "texto_y_figuras": "Texto y figuras",
        "tablas": "Tablas",
        "referencias": "Referencias",
        "solicitudes": "Qué se solicita a los autores",
        "evidencia": "Evidencia externa",
        "lectura": "Lectura",
        "estudios": "Estudios encontrados",
        "relevancia": "Relevancia",
        "pagina": "Página",
        "generado": "Generado el",
        "aviso": "Informe generado automáticamente con IA (EvidenciaMed). Debe ser verificado por el revisor antes de usarse.",
        "truncado": "Nota: el manuscrito excedía el máximo procesable y se recortó al final; esta revisión no cubre el texto omitido.",
    },
    "en": {
        "doc": "Critical manuscript review",
        "tipo": "Study type",
        "recomendacion": "Recommendation",
        "resumen": "Study data and summary",
        "diseno": "Design",
        "poblacion": "Population and data",
        "desenlace": "Outcome",
        "exposicion": "Exposure or intervention",
        "resultados": "Main results",
        "conclusion_autores": "Authors' conclusion",
        "evaluacion": "Overall assessment",
        "fortalezas": "Strengths",
        "mayores": "Major comments",
        "menores": "Minor comments",
        "texto_y_figuras": "Text and figures",
        "tablas": "Tables",
        "referencias": "References",
        "solicitudes": "Requests to the authors",
        "evidencia": "External evidence",
        "lectura": "Interpretation",
        "estudios": "Studies found",
        "relevancia": "Relevance",
        "pagina": "Page",
        "generado": "Generated on",
        "aviso": "Report generated automatically with AI (EvidenciaMed). It must be verified by the reviewer before use.",
        "truncado": "Note: the manuscript exceeded the maximum processable length and was cut at the end; this review does not cover the omitted text.",
    },
}

# Símbolos que Helvetica (WinAnsi) no tiene -> equivalente en texto
_REEMPLAZOS = {
    "≥": ">=", "≤": "<=", "≈": "~", "≠": "!=", "→": "->", "←": "<-", "↑": "^", "↓": "v",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "Δ": "Delta", "κ": "kappa",
    "λ": "lambda", "σ": "sigma", "χ": "chi", "π": "pi", "ρ": "rho", "τ": "tau",
    "θ": "theta", "ω": "omega", "μ": "µ",
    "−": "-", "‐": "-", "‑": "-", "⁻": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "": "", "‌": "", "‍": "", "﻿": "",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
}


def _limpiar(texto) -> str:
    """Deja el texto apto para fuentes estándar de PDF (Latin-1/cp1252)."""
    if texto is None:
        return ""
    t = str(texto)
    for origen, destino in _REEMPLAZOS.items():
        t = t.replace(origen, destino)
    t = "".join(ch for ch in t if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return t.encode("cp1252", "replace").decode("cp1252")


def _p(texto) -> str:
    """Texto limpio y escapado para usar dentro de un Paragraph."""
    return escape(_limpiar(texto))


def _s(valor) -> str:
    return "" if valor is None else str(valor).strip()


def _estilos() -> dict:
    base = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#1b1f27"), alignment=TA_LEFT)
    return {
        "titulo": ParagraphStyle("titulo", parent=None, fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceAfter=3),
        "sub": ParagraphStyle("sub", parent=None, fontName="Helvetica", fontSize=9.5, leading=13, textColor=GRAY, spaceAfter=2),
        "h1": ParagraphStyle("h1", parent=None, fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=13, spaceAfter=5, keepWithNext=1),
        "h2": ParagraphStyle("h2", parent=None, fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=7, spaceAfter=3, keepWithNext=1),
        "cuerpo": ParagraphStyle("cuerpo", parent=None, spaceAfter=5, **base),
        "item": ParagraphStyle("item", parent=None, leftIndent=16, bulletIndent=3, spaceAfter=3, **base),
        "mayor_t": ParagraphStyle("mayor_t", parent=None, fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.HexColor("#1b1f27"), spaceBefore=4, spaceAfter=1, keepWithNext=1),
        "mayor_d": ParagraphStyle("mayor_d", parent=None, leftIndent=16, spaceAfter=5, **base),
        "meta": ParagraphStyle("meta", parent=None, fontName="Helvetica", fontSize=8.5, leading=12, textColor=GRAY, leftIndent=0, spaceAfter=1),
        "nota": ParagraphStyle("nota", parent=None, fontName="Helvetica-Oblique", fontSize=8.5, leading=12, textColor=GRAY, spaceBefore=4, spaceAfter=4),
        "badge": ParagraphStyle("badge", parent=None, fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.white),
    }


def _color_recomendacion(rec: str):
    r = rec.lower()
    if "rechaz" in r or "reject" in r:
        return colors.HexColor("#b3261e")
    if "mayor" in r or "major" in r:
        return colors.HexColor("#c26a00")
    if "menor" in r or "minor" in r:
        return colors.HexColor("#1a56b0")
    if "acept" in r or "accept" in r:
        return colors.HexColor("#1e7b4b")
    return GRAY


def _lista(items, estilos, numerada=False):
    out = []
    for i, it in enumerate(items or [], start=1):
        txt = _s(it)
        if not txt:
            continue
        out.append(Paragraph(_p(txt), estilos["item"], bulletText=f"{i}." if numerada else "•"))
    return out


def review_pdf_filename(review: dict, lang: str = "es") -> str:
    """Nombre de archivo seguro, p. ej. revision_aspirin-and-heterotopic_es.pdf"""
    titulo = _s((review or {}).get("titulo")) or "informe"
    t = unicodedata.normalize("NFKD", titulo).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:50].strip("-") or "informe"
    return f"revision_{t}_{'en' if lang == 'en' else 'es'}.pdf"


def build_review_pdf(review: dict, lang: str = "es") -> bytes:
    """Construye el PDF del informe y devuelve sus bytes."""
    if not isinstance(review, dict) or not review:
        raise ValueError("Informe inválido: no se puede generar el PDF.")
    L = LABELS["en" if lang == "en" else "es"]
    E = _estilos()
    story = []

    # --- Encabezado
    story.append(Paragraph(_p(_s(review.get("titulo")) or L["doc"]), E["titulo"]))
    sub = " · ".join(x for x in (_s(review.get("revista_o_codigo")), _s(review.get("tipo_estudio"))) if x)
    if sub:
        story.append(Paragraph(_p(sub), E["sub"]))
    story.append(Paragraph(_p(f"{L['doc']} · {L['generado']} {date.today().isoformat()}"), E["sub"]))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.8, color=NAVY, spaceAfter=8))

    if review.get("texto_truncado"):
        story.append(Paragraph(_p(L["truncado"]), E["nota"]))

    # --- Recomendación
    rec = _s(review.get("recomendacion"))
    if rec:
        story.append(Paragraph(_p(L["recomendacion"]), E["h1"]))
        badge = Table([[Paragraph(_p(rec), E["badge"])]], hAlign="LEFT")
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _color_recomendacion(rec)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(badge)
        story.append(Spacer(1, 5))
    just = _s(review.get("justificacion_recomendacion"))
    if just:
        story.append(Paragraph(_p(just), E["cuerpo"]))

    # --- Datos y resumen del estudio
    rs = review.get("resumen_estudio") if isinstance(review.get("resumen_estudio"), dict) else {}
    campos = [("diseno", "diseno"), ("poblacion_y_datos", "poblacion"),
              ("desenlace", "desenlace"), ("exposicion_o_intervencion", "exposicion")]
    filas = [(L[lab], _s(rs.get(k))) for k, lab in campos if _s(rs.get(k))]
    resultados = rs.get("resultados_centrales") if isinstance(rs.get("resultados_centrales"), list) else []
    conclusion = _s(rs.get("conclusion_autores"))
    if filas or resultados or conclusion:
        story.append(Paragraph(_p(L["resumen"]), E["h1"]))
        for etiqueta, valor in filas:
            story.append(Paragraph(f"<b>{_p(etiqueta)}:</b> {_p(valor)}", E["cuerpo"]))
        if resultados:
            story.append(Paragraph(f"<b>{_p(L['resultados'])}:</b>", E["cuerpo"]))
            story.extend(_lista(resultados, E))
        if conclusion:
            story.append(Paragraph(f"<b>{_p(L['conclusion_autores'])}:</b> {_p(conclusion)}", E["cuerpo"]))

    # --- Evaluación general y fortalezas
    evaluacion = _s(review.get("evaluacion_general"))
    fortalezas = review.get("fortalezas") if isinstance(review.get("fortalezas"), list) else []
    if evaluacion or fortalezas:
        story.append(Paragraph(_p(L["evaluacion"]), E["h1"]))
        if evaluacion:
            story.append(Paragraph(_p(evaluacion), E["cuerpo"]))
        if fortalezas:
            story.append(Paragraph(_p(L["fortalezas"]), E["h2"]))
            story.extend(_lista(fortalezas, E))

    # --- Comentarios mayores
    mayores = review.get("comentarios_mayores") if isinstance(review.get("comentarios_mayores"), list) else []
    mayores = [c for c in mayores if isinstance(c, dict) and (_s(c.get("titulo")) or _s(c.get("detalle")))]
    if mayores:
        story.append(Paragraph(_p(L["mayores"]), E["h1"]))
        for i, c in enumerate(mayores, start=1):
            titulo = _s(c.get("titulo"))
            cab = f"{i}. {titulo}" if titulo else f"{i}."
            bloque = [Paragraph(_p(cab), E["mayor_t"])]
            if _s(c.get("detalle")):
                bloque.append(Paragraph(_p(c.get("detalle")), E["mayor_d"]))
            story.append(KeepTogether(bloque))

    # --- Comentarios menores
    menores = review.get("comentarios_menores") if isinstance(review.get("comentarios_menores"), dict) else {}
    secciones = [(k, menores.get(k)) for k in ("texto_y_figuras", "tablas", "referencias")
                 if isinstance(menores.get(k), list) and any(_s(x) for x in menores.get(k))]
    if secciones:
        story.append(Paragraph(_p(L["menores"]), E["h1"]))
        for k, items in secciones:
            story.append(Paragraph(_p(L[k]), E["h2"]))
            story.extend(_lista(items, E))

    # --- Solicitudes a los autores
    solicitudes = review.get("solicitudes_autores") if isinstance(review.get("solicitudes_autores"), list) else []
    if any(_s(x) for x in solicitudes):
        story.append(Paragraph(_p(L["solicitudes"]), E["h1"]))
        story.extend(_lista(solicitudes, E, numerada=True))

    # --- Evidencia externa
    ev = review.get("evidencia_externa") if isinstance(review.get("evidencia_externa"), dict) else None
    if ev:
        story.append(Paragraph(_p(L["evidencia"]), E["h1"]))
        if _s(ev.get("lectura")):
            story.append(Paragraph(f"<b>{_p(L['lectura'])}.</b> {_p(ev.get('lectura'))}", E["cuerpo"]))
        estudios = ev.get("estudios") if isinstance(ev.get("estudios"), list) else []
        estudios = [e for e in estudios if isinstance(e, dict)]
        if estudios:
            story.append(Paragraph(_p(L["estudios"]), E["h2"]))
            for e in estudios:
                titulo = _p(_s(e.get("titulo")) or _s(e.get("pmid")))
                url = _s(e.get("url"))
                if url.startswith("http"):
                    href = escape(url, {'"': "&quot;"})
                    titulo_html = f'<link href="{href}" color="{LINK}">{titulo}</link>'
                else:
                    titulo_html = titulo
                meta = " · ".join(x for x in (
                    _s(e.get("revista_anio")),
                    f"PMID {_s(e.get('pmid'))}" if _s(e.get("pmid")) else "",
                    f"{L['relevancia']}: {_s(e.get('relevancia'))}" if _s(e.get("relevancia")) else "",
                ) if x)
                bloque = [Paragraph(f"<b>{titulo_html}</b>", E["mayor_t"])]
                if meta:
                    bloque.append(Paragraph(_p(meta), ParagraphStyle("m2", parent=E["meta"], leftIndent=16)))
                if _s(e.get("hallazgo")):
                    bloque.append(Paragraph(_p(e.get("hallazgo")), E["mayor_d"]))
                story.append(KeepTogether(bloque))
        if _s(ev.get("alcance")):
            story.append(Paragraph(_p(ev.get("alcance")), E["nota"]))

    # --- Documento con pie de página
    def _pie(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        ancho, _alto = letter
        canvas.drawString(2 * cm, 1.2 * cm, _limpiar(L["aviso"]))
        canvas.drawRightString(ancho - 2 * cm, 1.2 * cm, f"{_limpiar(L['pagina'])} {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2.1 * cm,
        title=_limpiar(_s(review.get("titulo")) or L["doc"]), author="EvidenciaMed",
    )
    doc.build(story, onFirstPage=_pie, onLaterPages=_pie)
    return buf.getvalue()
      
