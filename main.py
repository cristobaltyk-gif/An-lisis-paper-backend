import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional

from modules.search import clean_doi, fetch_crossref, search_pubmed, search_scielo
from modules.analysis import build_content, analyze
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
    allow_methods=["POST", "GET", "DELETE"], allow_headers=["*"])

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
