import httpx
import fitz  # PyMuPDF
from typing import Optional
from modules.screener import score_paper

HEADERS = {"User-Agent": "EvidenciaMed/1.0 (mailto:contacto@cleversalud.cl)"}


def clean_doi(raw: str) -> str:
    """Normaliza cualquier formato de DOI."""
    doi = raw.strip()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


async def fetch_crossref(doi: str) -> dict:
    """Obtiene metadatos del paper desde CrossRef."""
    url = f"https://api.crossref.org/works/{doi}"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers=HEADERS)
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


async def fetch_unpaywall(doi: str) -> Optional[str]:
    """Intenta obtener texto completo desde PDF open access en Unpaywall."""
    url = f"https://api.unpaywall.org/v2/{doi}?email=contacto@cleversalud.cl"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers=HEADERS)
        if r.status_code != 200:
            return None
        data = r.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url:
            return None
        try:
            pr = await c.get(pdf_url, follow_redirects=True, timeout=30)
            if pr.status_code == 200 and "pdf" in pr.headers.get("content-type", ""):
                doc = fitz.open(stream=pr.content, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                return text[:15000]
        except Exception:
            return None
    return None


async def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """
    Busca papers en PubMed por término clínico.
    Devuelve lista rankeada con score calculado.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # 1. Buscar IDs
    async with httpx.AsyncClient(timeout=15) as c:
        search = await c.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }, headers=HEADERS)
        if search.status_code != 200:
            return []
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

    # 2. Obtener detalles via esummary
    async with httpx.AsyncClient(timeout=15) as c:
        summary = await c.get(f"{base}/esummary.fcgi", params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }, headers=HEADERS)
        if summary.status_code != 200:
            return []
        result_data = summary.json().get("result", {})

    # 3. Construir papers con score
    papers = []
    for pmid in ids:
        item = result_data.get(pmid, {})
        if not item:
            continue
        authors = ", ".join(
            a.get("name", "") for a in item.get("authors", [])[:3]
        )
        doi = next(
            (
                aid.get("value", "")
                for aid in item.get("articleids", [])
                if aid.get("idtype") == "doi"
            ),
            None,
        )
        paper = {
            "pmid": pmid,
            "title": item.get("title", ""),
            "authors": authors,
            "journal": item.get("source", ""),
            "year": item.get("pubdate", "")[:4],
            "doi": doi,
            "abstract": "",
            "open_access": doi is not None,
        }
        paper["score"] = score_paper(paper)
        papers.append(paper)

    # 4. Ordenar por score descendente
    return sorted(papers, key=lambda x: x["score"], reverse=True)
