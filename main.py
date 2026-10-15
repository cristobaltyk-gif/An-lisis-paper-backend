import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional

from modules.search import clean_doi, fetch_crossref, search_pubmed, search_scielo
from modules.analysis import (
    build_content, analyze,
    extract_text, review_manuscript, add_external_evidence, translate_review,
    EXTENSIONES_SOPORTADAS,
)
from modules.reporte_pdf import build_review_pdf, review_pdf_filename
from modules.downloader import get_fulltext
from modules.screener import run_screener, load_screener, STREAMS
from modules.memory import mark_as_read, get_all_read, clear_read
from modules.pacientes import router as pacientes_router
from modules.document_generator import generate_document, verify_document_key

async def scheduled_screener():
    while True:
        for stream in STREAMS:
            try:
                result = await run_screener(stream)
                print(f"[Scheduler] {stream}: {result['total']} papers")
            except Exception as e:
                print(f"[Scheduler] Error {stream}: {e}")
        await asyncio.sleep(7 * 24 * 60 * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(scheduled_screener())
    yield

app = FastAPI(title="EvidenciaMed API", version="4.0.0", lifespan=lifespan)
app.include_router(pacientes_router)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "DELETE"], allow_headers=["*"],
    # Para que el frontend lea el nombre del PDF descargado (/review/pdf)
    expose_headers=["Content-Disposition"])

API_KEY = os.getenv("EVIDENCIAMED_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_key(key: str = Security(api_key_header)):
    if not API_KEY or key != API_KEY:
        raise HTTPException(status_code=403, detail="API key inválida")
    return key

# Clave separada, exclusiva para el endpoint /generate/document.
# Solo la usa el backend de Examen Musculoesquelético (server-to-server),
# nunca el frontend de interrogadores directamente.
document_key_header = APIKeyHeader(name="X-Document-Key", auto_error=True)

async def verify_doc_key(key: str = Security(document_key_header)):
    if not verify_document_key(key):
        raise HTTPException(status_code=403, detail="Document key inválida")
    return key

class DoiRequest(BaseModel):
    doi: str

class TextRequest(BaseModel):
    text: str
    doi: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

class ReadRequest(BaseModel):
    pmid: str
    stream: str

class DocumentRequest(BaseModel):
    papers: list[dict]
    tema: str
    autor: Optional[str] = None

class ReviewRequest(BaseModel):
    review: dict

class ReviewPdfRequest(BaseModel):
    review: dict
    lang: str = "es"

@app.get("/health")
async def health():
    return {"status": "ok", "version": "4.0.0"}

@app.post("/analyze/doi", dependencies=[Depends(verify_key)])
async def analyze_doi(req: DoiRequest):
    doi = clean_doi(req.doi)
    meta = await fetch_crossref(doi)
    if not meta:
        raise HTTPException(status_code=404, detail=f"DOI no encontrado: {doi}")
    fulltext, fuente = await get_fulltext(doi=doi)
    content = build_content(meta, doi, fulltext, fuente)
    result = await analyze(content)
    result["doi"] = doi
    result["open_access"] = fulltext is not None
    result["fuente"] = fuente
    return result

@app.post("/analyze/text", dependencies=[Depends(verify_key)])
async def analyze_text(req: TextRequest):
    if len(req.text.strip()) < 100:
        raise HTTPException(status_code=422, detail="Texto demasiado corto.")
    content = req.text[:15000]
    if req.doi:
        content = f"DOI: {req.doi}\n\n{content}"
    result = await analyze(content)
    if req.doi:
        result["doi"] = req.doi
    result["fuente"] = "texto_manual"
    return result

@app.post("/search", dependencies=[Depends(verify_key)])
async def search_papers(req: SearchRequest):
    """
    Busca en PubMed y SciELO EN PARALELO, cada una hasta max_results
    (tope de 20 por fuente, no combinado). Se muestran TODOS los
    resultados de ambas fuentes juntos, ordenados por score de mayor
    a menor — sin recortar el total ni reservar cupos: si una fuente
    trae menos que el tope (o nada), simplemente no se rellena con
    más resultados de la otra.
    """
    if len(req.query.strip()) < 3:
        raise HTTPException(status_code=422, detail="Query demasiado corta.")

    max_results = min(req.max_results, 20)

    papers_pubmed, papers_scielo = await asyncio.gather(
        search_pubmed(req.query, max_results),
        search_scielo(req.query, max_results),
    )

    for p in papers_pubmed:
        p.setdefault("fuente", "pubmed")

    papers = sorted(
        papers_pubmed + papers_scielo,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    return {"query": req.query, "total": len(papers), "papers": papers}

@app.get("/screener/{stream}", dependencies=[Depends(verify_key)])
async def get_screener(stream: str, background_tasks: BackgroundTasks):
    if stream not in STREAMS:
        raise HTTPException(status_code=404, detail=f"Stream inválido.")
    data = load_screener(stream)
    if data is None:
        background_tasks.add_task(run_screener, stream)
        return {"stream": stream, "status": "generating",
                "message": "Generando. Reintenta en 60 segundos.",
                "papers": [], "total": 0}
    return data

@app.post("/screener/{stream}/refresh", dependencies=[Depends(verify_key)])
async def refresh_screener(stream: str, background_tasks: BackgroundTasks):
    if stream not in STREAMS:
        raise HTTPException(status_code=404, detail="Stream inválido.")
    background_tasks.add_task(run_screener, stream)
    return {"status": "refreshing", "stream": stream}

@app.post("/memory/read", dependencies=[Depends(verify_key)])
async def mark_read(req: ReadRequest):
    if req.stream not in STREAMS:
        raise HTTPException(status_code=404, detail="Stream inválido.")
    mark_as_read(req.stream, req.pmid)
    return {"status": "ok", "pmid": req.pmid, "stream": req.stream}

@app.get("/memory/{stream}", dependencies=[Depends(verify_key)])
async def get_read(stream: str):
    if stream not in STREAMS:
        raise HTTPException(status_code=404, detail="Stream inválido.")
    return {"stream": stream, "read": get_all_read(stream)}

@app.delete("/memory/{stream}", dependencies=[Depends(verify_key)])
async def clear_stream_read(stream: str):
    if stream not in STREAMS:
        raise HTTPException(status_code=404, detail="Stream inválido.")
    clear_read(stream)
    return {"status": "ok", "stream": stream}

@app.post("/generate/document", dependencies=[Depends(verify_doc_key)])
async def generate_document_endpoint(req: DocumentRequest):
    if not req.papers:
        raise HTTPException(status_code=422, detail="Se requiere al menos un paper analizado.")
    if len(req.tema.strip()) < 3:
        raise HTTPException(status_code=422, detail="Tema demasiado corto.")
    try:
        resultado = await generate_document(req.papers, req.tema.strip(), req.autor or "")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return resultado


# ---------------------------------------------------------------------------
# REVISIÓN CRÍTICA DE MANUSCRITO SUBIDO (modo "Subir archivo")
#
#   POST /review/file       -> sube PDF/DOCX/TXT; responde en streaming (SSE)
#                              con el progreso por etapas y, al final, el informe
#   POST /review/translate  -> traduce al inglés un informe ya aprobado
#   POST /review/pdf        -> genera el PDF de un informe (es | en)
#
# El servidor no guarda nada: el frontend conserva el informe y lo reenvía
# para traducir o descargar el PDF.
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB
EVIDENCIA_RESULTADOS_POR_CONSULTA = 5
EVIDENCIA_MAX_TEXTOS = 5
EVIDENCIA_TIMEOUT_TEXTO = 45          # segundos por estudio en la cascada de texto completo
LATIDO_SEGUNDOS = 15                  # mantiene viva la conexión durante llamadas largas


def _sse(evento: dict) -> str:
    return f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"


def _progreso(etapa: str, pct: int, mensaje: str) -> str:
    return _sse({"type": "progress", "stage": etapa, "pct": pct, "message": mensaje})


async def _latidos(task: asyncio.Task):
    """Mientras la tarea corre, emite un comentario SSE cada LATIDO_SEGUNDOS
    para que proxies y navegador no corten la conexión por inactividad."""
    while not task.done():
        await asyncio.wait({task}, timeout=LATIDO_SEGUNDOS)
        if not task.done():
            yield ": latido\n\n"


async def _texto_estudio(paper: dict) -> str:
    """Texto disponible de un estudio de PubMed vía la cascada de texto completo.
    Devuelve '' si no se encuentra o si tarda demasiado."""
    try:
        texto, _fuente = await asyncio.wait_for(
            get_fulltext(doi=paper.get("doi") or None, pmid=paper.get("pmid") or None),
            timeout=EVIDENCIA_TIMEOUT_TEXTO,
        )
        return texto or ""
    except Exception:
        return ""


async def _leer_textos(papers: list[dict]) -> list[str]:
    """Obtiene en paralelo el texto disponible de cada estudio."""
    return list(await asyncio.gather(*(_texto_estudio(p) for p in papers)))


async def _buscar_estudios(terminos: list[str]) -> list[dict]:
    """Busca en PubMed cada término, quita duplicados por PMID y ordena por score."""
    resultados = await asyncio.gather(
        *(search_pubmed(t, EVIDENCIA_RESULTADOS_POR_CONSULTA) for t in terminos),
        return_exceptions=True,
    )
    por_pmid = {}
    for lista in resultados:
        if isinstance(lista, Exception):
            continue
        for p in lista:
            pmid = str(p.get("pmid") or "")
            if pmid and pmid not in por_pmid:
                por_pmid[pmid] = p
    return sorted(por_pmid.values(), key=lambda x: x.get("score", 0), reverse=True)


@app.post("/review/file", dependencies=[Depends(verify_key)])
async def review_file(file: UploadFile = File(...)):
    nombre = file.filename or ""
    if not nombre.lower().endswith(EXTENSIONES_SOPORTADAS):
        raise HTTPException(status_code=422, detail="Formato no soportado. Sube un PDF, un Word (.docx) o un .txt.")
    # Se lee completo aquí, antes de empezar el stream
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el máximo de 25 MB.")
    if not data:
        raise HTTPException(status_code=422, detail="El archivo está vacío.")

    # 'estado' guarda la tarea en curso para poder cancelarla si el cliente se va
    estado = {"tarea": None}

    async def correr(coro):
        """Lanza la corrutina como tarea, emite latidos mientras corre y deja
        el resultado en estado['resultado'] (o propaga su excepción)."""
        estado["tarea"] = asyncio.create_task(coro)
        async for latido in _latidos(estado["tarea"]):
            yield latido
        tarea, estado["tarea"] = estado["tarea"], None
        estado["resultado"] = tarea.result()

    async def pasos():
        # 1. Lectura del archivo
        yield _progreso("lectura", 5, "Leyendo el archivo...")
        try:
            texto = await asyncio.to_thread(extract_text, nombre, data)
        except ValueError as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        # 2. Revisión crítica (la etapa más larga)
        yield _progreso("revision", 15, "Revisando el manuscrito completo. Esta etapa tarda unos minutos...")
        try:
            async for latido in correr(review_manuscript(texto)):
                yield latido
            review = estado["resultado"]
        except ValueError as e:
            yield _sse({"type": "error", "message": str(e)})
            return
        except Exception as e:
            print(f"[review/file] Error en revisión: {e}")
            yield _sse({"type": "error", "message": "No se pudo completar la revisión. Intenta de nuevo en unos minutos."})
            return

        # 3. Evidencia externa (si falla, el informe sigue sin esta sección)
        try:
            yield _progreso("busqueda", 60, "Buscando estudios relacionados en PubMed...")
            async for latido in correr(_buscar_estudios(review.get("terminos_busqueda", []))):
                yield latido
            estudios = estado["resultado"]

            a_leer = estudios[:EVIDENCIA_MAX_TEXTOS]
            yield _progreso("lectura_estudios", 70, f"Leyendo {len(a_leer)} de {len(estudios)} estudios encontrados...")
            async for latido in correr(_leer_textos(a_leer)):
                yield latido
            for p, t in zip(a_leer, estado["resultado"]):
                p["texto"] = t

            yield _progreso("evidencia", 85, "Contrastando el manuscrito con la evidencia encontrada...")
            async for latido in correr(add_external_evidence(review, estudios)):
                yield latido
            review = estado["resultado"]
        except Exception as e:
            print(f"[review/file] Error en evidencia externa: {e}")
            review["evidencia_externa"] = {
                "lectura": "",
                "estudios": [],
                "alcance": "No se pudo completar la búsqueda de evidencia externa; el informe no incluye esta sección.",
            }

        # 4. Listo
        yield _progreso("listo", 100, "Informe listo.")
        yield _sse({"type": "result", "review": review})

    async def generar():
        try:
            async for fragmento in pasos():
                yield fragmento
            yield "data: [DONE]\n\n"
        finally:
            # Si el cliente se desconecta a mitad de camino, no dejar trabajo colgado.
            # (Sin yield aquí: un generador que se está cerrando no puede emitir.)
            tarea = estado["tarea"]
            if tarea is not None and not tarea.done():
                tarea.cancel()

    return StreamingResponse(
        generar(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/review/translate", dependencies=[Depends(verify_key)])
async def review_translate(req: ReviewRequest):
    if not req.review:
        raise HTTPException(status_code=422, detail="Falta el informe a traducir.")
    try:
        traducido = await translate_review(req.review)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        print(f"[review/translate] Error: {e}")
        raise HTTPException(status_code=502, detail="No se pudo traducir el informe. Intenta de nuevo.")
    return {"review": traducido}


@app.post("/review/pdf", dependencies=[Depends(verify_key)])
async def review_pdf(req: ReviewPdfRequest):
    if req.lang not in ("es", "en"):
        raise HTTPException(status_code=422, detail="Idioma inválido (usa 'es' o 'en').")
    try:
        pdf = await asyncio.to_thread(build_review_pdf, req.review, req.lang)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    nombre = review_pdf_filename(req.review, req.lang)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
    )
    
