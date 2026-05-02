import os
import httpx
import fitz  # PyMuPDF
import anthropic
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI(title="EvidenciaMed API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── AUTH ──────────────────────────────────────────────────────────
API_KEY = os.getenv("EVIDENCIAMED_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_key(key: str = Security(api_key_header)):
    if not API_KEY or key != API_KEY:
        raise HTTPException(status_code=403, detail="API key inválida")
    return key

# ── ANTHROPIC ─────────────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

SYSTEM_PROMPT = """Eres un experto en medicina basada en evidencia y metodología de investigación científica clínica.
Analiza el artículo científico y devuelve EXCLUSIVAMENTE un JSON válido con esta estructura exacta, sin texto adicional:

{
  "titulo": "título del paper",
  "autores": "autores principales",
  "revista": "nombre de la revista y año",
  "doi": "DOI del artículo",
  "tipo_estudio": "tipo de diseño metodológico",
  "nivel_evidencia_oxford": "1a/1b/2a/2b/3a/3b/4/5",
  "nivel_evidencia_descripcion": "descripción del nivel Oxford",
  "grado_recomendacion": "A/B/C/D",
  "calidad_grade": "Alta/Moderada/Baja/Muy baja",
  "pico": {
    "poblacion": "descripción de la población estudiada",
    "intervencion": "intervención principal",
    "comparador": "grupo control o comparador",
    "outcome": "desenlaces principales medidos"
  },
  "resumen_ejecutivo": "resumen en 3-4 oraciones claras del hallazgo principal",
  "hallazgos_clave": ["hallazgo 1", "hallazgo 2", "hallazgo 3"],
  "tamano_muestra": "N total y características",
  "seguimiento": "duración del seguimiento",
  "limitaciones": ["limitación 1", "limitación 2", "limitación 3"],
  "aplicabilidad_clinica": "cómo aplicar estos resultados en práctica clínica",
  "aplicabilidad_chile": "relevancia específica para el contexto chileno/latinoamericano",
  "conclusion_critica": "evaluación crítica honesta de fortalezas y debilidades",
  "puntuacion_calidad": 75,
  "categoria": "Terapéutica/Diagnóstico/Pronóstico/Etiología/Revisión/Meta-análisis/Guía Clínica"
}"""

# ── MODELOS ───────────────────────────────────────────────────────
class DoiRequest(BaseModel):
    doi: str

class TextRequest(BaseModel):
    text: str
    doi: Optional[str] = None

# ── HELPERS ───────────────────────────────────────────────────────
async def fetch_crossref_metadata(doi: str) -> dict:
    """Obtiene metadatos del paper desde CrossRef."""
    url = f"https://api.crossref.org/works/{doi}"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers={"User-Agent": "EvidenciaMed/1.0 (mailto:contacto@clevercalud.cl)"})
        if r.status_code != 200:
            return {}
        data = r.json().get("message", {})
        return {
            "title": data.get("title", [""])[0],
            "authors": ", ".join(
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in data.get("author", [])[:4]
            ),
            "journal": data.get("container-title", [""])[0],
            "year": str(data.get("published", {}).get("date-parts", [[""]])[0][0]),
            "abstract": data.get("abstract", ""),
            "url": data.get("URL", ""),
        }

async def fetch_fulltext_unpaywall(doi: str) -> Optional[str]:
    """Intenta obtener PDF open access desde Unpaywall."""
    url = f"https://api.unpaywall.org/v2/{doi}?email=contacto@cleversalud.cl"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url)
        if r.status_code != 200:
            return None
        data = r.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url:
            return None
        # Descargar PDF
        try:
            pr = await c.get(pdf_url, follow_redirects=True, timeout=30)
            if pr.status_code == 200 and "pdf" in pr.headers.get("content-type", ""):
                doc = fitz.open(stream=pr.content, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                return text[:15000]  # Limitar tokens
        except Exception:
            return None
    return None

async def call_anthropic(content: str) -> dict:
    """Llama a Claude y devuelve el JSON parseado."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}]
    )
    raw = message.content[0].text
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

# ── ENDPOINTS ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "EvidenciaMed API"}

@app.post("/analyze/doi", dependencies=[Depends(verify_key)])
async def analyze_doi(req: DoiRequest):
    """Recibe un DOI, busca el paper y lo analiza."""
    doi = req.doi.strip().lstrip("https://doi.org/").lstrip("doi:")

    # 1. Metadatos CrossRef
    meta = await fetch_crossref_metadata(doi)
    if not meta:
        raise HTTPException(status_code=404, detail=f"DOI no encontrado en CrossRef: {doi}")

    # 2. Texto completo (Unpaywall) o solo abstract
    fulltext = await fetch_fulltext_unpaywall(doi)

    if fulltext and len(fulltext) > 500:
        content = f"""DOI: {doi}
Título: {meta.get('title','')}
Autores: {meta.get('authors','')}
Revista: {meta.get('journal','')} ({meta.get('year','')})

TEXTO COMPLETO:
{fulltext}"""
    elif meta.get("abstract"):
        content = f"""DOI: {doi}
Título: {meta.get('title','')}
Autores: {meta.get('authors','')}
Revista: {meta.get('journal','')} ({meta.get('year','')})

ABSTRACT:
{meta.get('abstract','')}

Nota: Solo se dispone del abstract. Analiza con la información disponible."""
    else:
        raise HTTPException(
            status_code=422,
            detail="No se pudo obtener el texto del paper. Puede ser un artículo de pago sin versión open access."
        )

    result = await call_anthropic(content)
    # Asegurar que el DOI quede en el resultado
    result["doi"] = doi
    result["open_access"] = fulltext is not None
    result["fuente"] = "fulltext" if (fulltext and len(fulltext) > 500) else "abstract"
    return result

@app.post("/analyze/text", dependencies=[Depends(verify_key)])
async def analyze_text(req: TextRequest):
    """Recibe texto libre (abstract o paper) y lo analiza."""
    if len(req.text.strip()) < 100:
        raise HTTPException(status_code=422, detail="El texto es demasiado corto para analizar.")
    content = req.text[:15000]
    if req.doi:
        content = f"DOI: {req.doi}\n\n{content}"
    result = await call_anthropic(content)
    if req.doi:
        result["doi"] = req.doi
    result["fuente"] = "texto_manual"
    return result
