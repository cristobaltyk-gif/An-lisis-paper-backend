import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional

from modules.search import clean_doi, fetch_crossref, fetch_unpaywall, search_pubmed
from modules.analysis import build_content, analyze

# ── APP ───────────────────────────────────────────────────────────
app = FastAPI(title="EvidenciaMed API", version="2.0.0")

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

# ── ENDPOINTS ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "EvidenciaMed API", "version": "2.0.0"}


@app.post("/analyze/doi", dependencies=[Depends(verify_key)])
async def analyze_doi(req: DoiRequest):
    """Recibe un DOI, busca el paper en CrossRef + Unpaywall y lo analiza."""
    doi = clean_doi(req.doi)

    meta = await fetch_crossref(doi)
    if not meta:
        raise HTTPException(status_code=404, detail=f"DOI no encontrado en CrossRef: {doi}")

    fulltext = await fetch_unpaywall(doi)
    content = build_content(meta, doi, fulltext)

    result = await analyze(content)
    result["doi"] = doi
    result["open_access"] = fulltext is not None
    result["fuente"] = "fulltext" if (fulltext and len(fulltext) > 500) else "abstract"
    return result


@app.post("/analyze/text", dependencies=[Depends(verify_key)])
async def analyze_text(req: TextRequest):
    """Recibe texto libre y lo analiza."""
    if len(req.text.strip()) < 100:
        raise HTTPException(status_code=422, detail="El texto es demasiado corto para analizar.")
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
    """Busca papers en PubMed por término clínico."""
    if len(req.query.strip()) < 3:
        raise HTTPException(status_code=422, detail="Query demasiado corta.")
    papers = await search_pubmed(req.query, min(req.max_results, 20))
    return {"query": req.query, "total": len(papers), "papers": papers}
        
