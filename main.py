import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional

from modules.search import clean_doi, fetch_crossref, fetch_unpaywall, search_pubmed
from modules.analysis import build_content, analyze
from modules.screener import run_screener, load_screener, STREAMS

# ── Scheduler ─────────────────────────────────────────────────────
async def scheduled_screener():
    while True:
        print("[Scheduler] Iniciando screener automático...")
        for stream in STREAMS:
            try:
                result = await run_screener(stream)
                print(f"[Scheduler] {stream}: {result['total']} papers rankeados")
            except Exception as e:
                print(f"[Scheduler] Error en {stream}: {e}")
        print("[Scheduler] Próxima ejecución en 7 días.")
        await asyncio.sleep(7 * 24 * 60 * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(scheduled_screener())
    yield

# ── APP ───────────────────────────────────────────────────────────
app = FastAPI(title="EvidenciaMed API", version="3.0.0", lifespan=lifespan)

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

# ── MODELOS ───────────────────────────────────────────────────────
class DoiRequest(BaseModel):
    doi: str

class TextRequest(BaseModel):
    text: str
    doi: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

# ── ENDPOINTS GENERALES ───────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "EvidenciaMed API", "version": "3.0.0"}

@app.post("/analyze/doi", dependencies=[Depends(verify_key)])
async def analyze_doi(req: DoiRequest):
    doi = clean_doi(req.doi)
    meta = await fetch_crossref(doi)
    if not meta:
        raise HTTPException(status_code=404, detail=f"DOI no encontrado: {doi}")
    fulltext = await fetch_unpaywall(doi)
    content = build_content(meta, doi, fulltext)
    result = await analyze(content)
    result["doi"] = doi
    result["open_access"] = fulltext is not None
    result["fuente"] = "fulltext" if (fulltext and len(fulltext) > 500) else "abstract"
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
    if len(req.query.strip()) < 3:
        raise HTTPException(status_code=422, detail="Query demasiado corta.")
    papers = await search_pubmed(req.query, min(req.max_results, 20))
    return {"query": req.query, "total": len(papers), "papers": papers}

# ── ENDPOINTS SCREENER ────────────────────────────────────────────
@app.get("/screener/{stream}", dependencies=[Depends(verify_key)])
async def get_screener(stream: str, background_tasks: BackgroundTasks):
    if stream not in STREAMS:
        raise HTTPException(status_code=404, detail=f"Stream inválido. Usa: {list(STREAMS.keys())}")
    data = load_screener(stream)
    if data is None:
        background_tasks.add_task(run_screener, stream)
        return {
            "stream": stream,
            "status": "generating",
            "message": "Generando resultados por primera vez. Reintenta en 60 segundos.",
            "papers": [],
            "total": 0,
        }
    return data

@app.post("/screener/{stream}/refresh", dependencies=[Depends(verify_key)])
async def refresh_screener(stream: str, background_tasks: BackgroundTasks):
    if stream not in STREAMS:
        raise HTTPException(status_code=404, detail=f"Stream inválido. Usa: {list(STREAMS.keys())}")
    background_tasks.add_task(run_screener, stream)
    return {"status": "refreshing", "stream": stream, "message": "Actualizando en background..."}

@app.get("/screener", dependencies=[Depends(verify_key)])
async def get_all_screeners():
    result = {}
    for stream in STREAMS:
        data = load_screener(stream)
        if data:
            result[stream] = {
                "total": data["total"],
                "updated_at": data["updated_at"],
                "top3": data["papers"][:3],
            }
        else:
            result[stream] = {"total": 0, "updated_at": None, "top3": []}
    return result
