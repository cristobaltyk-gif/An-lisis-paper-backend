import os
import json
import asyncio
import httpx
from datetime import datetime, timezone
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────
DATA_DIR = Path("/tmp/screener")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "EvidenciaMed/1.0 (mailto:contacto@cleversalud.cl)"}

STREAMS = {
    "cadera": [
        "total hip arthroplasty outcomes",
        "hip replacement implant selection",
        "femoral stem fixation cementless",
        "total hip arthroplasty complications",
        "hip arthroplasty bearing surface",
    ],
    "rodilla": [
        "total knee arthroplasty outcomes",
        "knee replacement implant selection",
        "tibial component fixation",
        "total knee arthroplasty complications",
        "knee arthroplasty patient reported outcomes",
    ],
}

# Revistas top — mayor puntaje
TOP_JOURNALS = {
    "new england journal of medicine": 10,
    "nejm": 10,
    "lancet": 10,
    "jama": 10,
    "journal of bone and joint surgery": 8,
    "jbjs": 8,
    "clinical orthopaedics and related research": 7,
    "corr": 7,
    "bone joint j": 7,
    "knee surgery sports traumatology arthroscopy": 6,
    "journal of arthroplasty": 6,
    "acta orthopaedica": 5,
    "orthopaedics traumatology surgery research": 5,
}

STUDY_KEYWORDS = {
    "randomized": 25, "randomised": 25, "rct": 25,
    "meta-analysis": 22, "meta analysis": 22, "systematic review": 20,
    "prospective": 12, "cohort": 10,
    "retrospective": 5, "case series": 3,
}


# ── Scoring ───────────────────────────────────────────────────────
def score_paper(paper: dict) -> int:
    score = 0

    # Año (max 30)
    try:
        year = int(paper.get("year", 0))
        current = datetime.now().year
        if year >= current:       score += 30
        elif year >= current - 1: score += 25
        elif year >= current - 2: score += 18
        elif year >= current - 3: score += 10
        else:                     score += 3
    except Exception:
        pass

    # Tipo de estudio desde título+abstract (max 25)
    text_lower = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    best_study = 0
    for kw, pts in STUDY_KEYWORDS.items():
        if kw in text_lower:
            best_study = max(best_study, pts)
    score += best_study

    # Open access (15)
    if paper.get("open_access"):
        score += 15

    # Revista (max 10)
    journal_lower = paper.get("journal", "").lower()
    for journal, pts in TOP_JOURNALS.items():
        if journal in journal_lower:
            score += pts
            break

    # DOI disponible (5)
    if paper.get("doi"):
        score += 5

    # N pacientes desde título (max 15)
    import re
    nums = re.findall(r'\b(\d{3,6})\b', paper.get("title", ""))
    if nums:
        max_n = max(int(n) for n in nums)
        if max_n >= 1000: score += 15
        elif max_n >= 500: score += 10
        elif max_n >= 100: score += 5

    return min(score, 100)


# ── PubMed fetch ──────────────────────────────────────────────────
async def fetch_pubmed_query(query: str, max_results: int = 20) -> list[dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    papers = []

    async with httpx.AsyncClient(timeout=20) as c:
        # Search
        search = await c.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json",
            "sort": "relevance",
            "datetype": "pdat", "reldate": 730,  # últimos 2 años
        }, headers=HEADERS)
        if search.status_code != 200:
            return []
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Summary
        summary = await c.get(f"{base}/esummary.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
        }, headers=HEADERS)
        if summary.status_code != 200:
            return []
        result_data = summary.json().get("result", {})

    for pmid in ids:
        item = result_data.get(pmid, {})
        if not item:
            continue
        authors = ", ".join(a.get("name", "") for a in item.get("authors", [])[:3])
        doi = next((
            aid.get("value", "") for aid in item.get("articleids", [])
            if aid.get("idtype") == "doi"
        ), None)

        papers.append({
            "pmid": pmid,
            "title": item.get("title", ""),
            "authors": authors,
            "journal": item.get("source", ""),
            "year": item.get("pubdate", "")[:4],
            "doi": doi,
            "abstract": "",
            "open_access": False,
        })

    return papers


async def check_unpaywall(doi: str) -> bool:
    """Verifica si el paper tiene versión open access."""
    if not doi:
        return False
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                f"https://api.unpaywall.org/v2/{doi}?email=contacto@cleversalud.cl",
                headers=HEADERS
            )
            if r.status_code == 200:
                return r.json().get("is_oa", False)
    except Exception:
        pass
    return False


# ── Motor principal ───────────────────────────────────────────────
async def run_screener(stream: str) -> dict:
    """
    Corre el screener para un stream (cadera o rodilla).
    Devuelve dict con papers rankeados.
    """
    queries = STREAMS.get(stream, [])
    all_papers: dict[str, dict] = {}

    # Recolectar papers de todas las queries
    for query in queries:
        papers = await fetch_pubmed_query(query, max_results=15)
        for p in papers:
            pmid = p["pmid"]
            if pmid not in all_papers:
                all_papers[pmid] = p

    # Verificar OA para los que tienen DOI (en paralelo, max 10)
    dois = [(pmid, p["doi"]) for pmid, p in all_papers.items() if p.get("doi")]
    dois = dois[:10]  # limitar llamadas

    async def set_oa(pmid, doi):
        all_papers[pmid]["open_access"] = await check_unpaywall(doi)

    await asyncio.gather(*[set_oa(pmid, doi) for pmid, doi in dois])

    # Puntuar y ordenar
    for pmid, paper in all_papers.items():
        paper["score"] = score_paper(paper)

    ranked = sorted(all_papers.values(), key=lambda x: x["score"], reverse=True)

    result = {
        "stream": stream,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(ranked),
        "papers": ranked,
    }

    # Guardar en disco
    path = DATA_DIR / f"{stream}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def load_screener(stream: str) -> dict | None:
    """Carga resultados guardados del screener."""
    path = DATA_DIR / f"{stream}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
      
