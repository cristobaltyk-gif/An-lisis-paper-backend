# modules/pacientes.py
import os
import asyncio
import json
import anthropic
from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules.search import search_pubmed
from modules.analysis import build_content, analyze
from modules.downloader import get_fulltext
from modules.search import fetch_crossref, clean_doi

router = APIRouter(prefix="/pacientes", tags=["pacientes"])

API_KEY = os.getenv("EVIDENCIAMED_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_key(key: str = Security(api_key_header)):
    if not API_KEY or key != API_KEY:
        raise HTTPException(status_code=403, detail="API key inválida")
    return key


SYSTEM_PROMPT = """Eres un médico del Instituto de Cirugía Articular (ICA) de Chile explicándole a un paciente qué dice la ciencia sobre su condición.

Recibirás análisis reales de papers científicos recientes sobre el tema. Tu trabajo es traducir esa evidencia a lenguaje simple y empático.

ESTRUCTURA DE RESPUESTA (usa exactamente estos títulos):

### ¿Qué es esto?
Explica la condición en 2-3 oraciones simples, sin jerga médica.

### ¿Qué dice la ciencia?
Basándote SOLO en los papers que recibes como contexto, explica:
- Qué tratamientos tienen evidencia
- Qué funciona mejor según los estudios
- Menciona el tipo de estudio cuando sea relevante (ej: "un estudio con 500 pacientes mostró...")

### Lo que puedes hacer
Lista concreta de acciones: en casa, ejercicios, cambios de hábito.

### Cuándo consultar al médico
- Señales de alarma claras
- Qué tipo de especialista buscar

### Mitos frecuentes
Desmiente 1-2 mitos comunes sobre esta condición.

### Recuerda
Cierra con: esta información es educativa y no reemplaza la consulta médica. Si estás en Chile, el equipo de ICA puede orientarte.

REGLAS:
- Lenguaje simple, cálido, como un médico de confianza
- NO inventes datos que no estén en los papers recibidos
- Si los papers no cubren algo relevante, dilo honestamente
- Responde siempre en español
- Máximo 500 palabras en total"""


class PatientRequest(BaseModel):
    query: str


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY no configurada.")
    return anthropic.Anthropic(api_key=api_key)


async def transform_query_to_pubmed(patient_query: str, client: anthropic.Anthropic) -> str:
    """
    Usa Claude para transformar la pregunta en lenguaje de paciente
    a una query optimizada para PubMed en inglés.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system="""Eres un experto en búsqueda de literatura médica.
Tu única tarea es convertir la pregunta de un paciente en una query corta y precisa para buscar en PubMed.

REGLAS:
- Responde SOLO con la query, sin explicaciones ni puntuación extra
- Máximo 6 palabras clave en inglés
- Usa terminología médica estándar
- Enfócate en el tema clínico principal, ignora el lenguaje coloquial

Ejemplos:
"me duele la rodilla al bajar escaleras" → "knee pain stairs osteoarthritis treatment"
"artrosis de cadera me opero?" → "hip osteoarthritis surgical indication arthroplasty"
"ácido hialurónico sirve para artrosis?" → "hyaluronic acid injection knee osteoarthritis efficacy"
"cuándo debo ponerme prótesis de rodilla" → "total knee replacement indication criteria timing"
""",
        messages=[{"role": "user", "content": patient_query}]
    )
    return response.content[0].text.strip()


async def analyze_paper_safe(paper: dict) -> dict | None:
    """Analiza un paper individual, retorna None si falla."""
    try:
        doi = paper.get("doi")
        fulltext, fuente = None, "abstract"

        if doi:
            doi_clean = clean_doi(doi)
            meta = await fetch_crossref(doi_clean)
            fulltext, fuente = await get_fulltext(doi=doi_clean)
            if meta:
                content = build_content(meta, doi_clean, fulltext, fuente)
            else:
                content = f"Title: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nJournal: {paper.get('journal', '')} ({paper.get('year', '')})"
        else:
            content = f"Title: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nJournal: {paper.get('journal', '')} ({paper.get('year', '')})\nPMID: {paper.get('pmid', '')}"

        result = await analyze(content)
        result["fuente"] = fuente
        return result
    except Exception as e:
        print(f"[Pacientes] Error analizando paper {paper.get('pmid')}: {e}")
        return None


def build_context_from_analyses(analyses: list[dict]) -> str:
    """Construye el contexto con los análisis para Claude."""
    parts = []
    for i, a in enumerate(analyses, 1):
        parts.append(f"""
--- PAPER {i} ---
Título: {a.get('titulo', 'N/A')}
Tipo de estudio: {a.get('tipo_estudio', 'N/A')}
Nivel de evidencia Oxford: {a.get('nivel_evidencia_oxford', 'N/A')}
Calidad GRADE: {a.get('calidad_grade', 'N/A')}
Puntuación de calidad: {a.get('puntuacion_calidad', 'N/A')}/100
Población: {a.get('pico', {}).get('poblacion', 'N/A')}
Intervención: {a.get('pico', {}).get('intervencion', 'N/A')}
Comparador: {a.get('pico', {}).get('comparador', 'N/A')}
Outcome: {a.get('pico', {}).get('outcome', 'N/A')}
Resumen ejecutivo: {a.get('resumen_ejecutivo', 'N/A')}
Hallazgos clave: {'; '.join(a.get('hallazgos_clave', []))}
Tamaño muestra: {a.get('tamano_muestra', 'N/A')}
Seguimiento: {a.get('seguimiento', 'N/A')}
""")
    return "\n".join(parts)


@router.post("/chat", dependencies=[Depends(verify_key)])
async def pacientes_chat(req: PatientRequest):
    """
    Pipeline completo para pacientes:
    1. Claude transforma pregunta del paciente a query PubMed
    2. Busca papers en PubMed
    3. Analiza los top 3 por score
    4. Claude responde en lenguaje simple basado en la evidencia
    """
    query = req.query.strip()
    if len(query) < 3:
        raise HTTPException(status_code=422, detail="Query demasiado corta.")

    client = get_client()

    # 1. Transformar query del paciente a query PubMed
    try:
        pubmed_query = await transform_query_to_pubmed(query, client)
        print(f"[Pacientes] Query original: '{query}' → PubMed: '{pubmed_query}'")
    except Exception as e:
        pubmed_query = query  # fallback a query original
        print(f"[Pacientes] Error transformando query, usando original: {e}")

    # 2. Buscar papers
    try:
        papers = await search_pubmed(pubmed_query, max_results=10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buscando papers: {e}")

    if not papers:
        raise HTTPException(status_code=404, detail="No se encontraron papers para esta consulta.")

    # 3. Ordenar por score y tomar top 3
    papers_sorted = sorted(papers, key=lambda p: p.get("score", 0), reverse=True)
    top_papers = papers_sorted[:3]

    # 4. Analizar los 3 en paralelo
    analyses_raw = await asyncio.gather(*[analyze_paper_safe(p) for p in top_papers])
    analyses = [a for a in analyses_raw if a is not None]

    if not analyses:
        raise HTTPException(status_code=500, detail="No se pudo analizar ningún paper.")

    analyses.sort(key=lambda a: a.get("puntuacion_calidad", 0), reverse=True)

    # 5. Construir contexto y responder en streaming
    context = build_context_from_analyses(analyses)
    user_message = f"""El paciente pregunta: {query}

Aquí están los {len(analyses)} papers más relevantes analizados por nuestro sistema:

{context}

Explícale al paciente en lenguaje simple qué dice esta evidencia científica sobre su pregunta."""

    papers_meta = [
        {
            "titulo": a.get("titulo", ""),
            "tipo_estudio": a.get("tipo_estudio", ""),
            "nivel_evidencia_oxford": a.get("nivel_evidencia_oxford", ""),
            "calidad_grade": a.get("calidad_grade", ""),
            "puntuacion_calidad": a.get("puntuacion_calidad", 0),
            "doi": a.get("doi", ""),
        }
        for a in analyses
    ]

    def generate():
        meta_chunk = json.dumps({"type": "papers_meta", "papers": papers_meta}, ensure_ascii=False)
        yield f"data: {meta_chunk}\n\n"

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                chunk = json.dumps({"type": "text", "text": text}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
