import os
import json
import asyncio
import httpx
import fitz  # PyMuPDF
import anthropic
from typing import Optional
from modules.screener import score_paper

HEADERS = {"User-Agent": "EvidenciaMed/1.0 (mailto:contacto@cleversalud.cl)"}

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def clean_doi(raw: str) -> str:
    """Normaliza cualquier formato de DOI."""
    doi = raw.strip()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


async def generate_search_variants(query: str, n: int = 4) -> list[str]:
    """
    A partir del término de búsqueda del usuario (en cualquier idioma), genera
    varias variantes de búsqueda en inglés con terminología médica/MeSH
    estándar y sinónimos clínicos relevantes — no solo una traducción directa,
    sino distintos ángulos del mismo tema (término más amplio, más específico,
    sinónimo aceptado), para ampliar la cobertura de la búsqueda en PubMed.
    Si falla, devuelve una lista con solo el término original (fallback seguro).
    """
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=250,
            system=(
                "Eres un experto en búsqueda bibliográfica médica (PubMed/MeSH). "
                f"A partir del término de búsqueda del usuario, genera {n} variantes de "
                "búsqueda en inglés, usando terminología médica/MeSH estándar y sinónimos "
                "clínicos relevantes (ej: término MeSH directo, sinónimo clínico aceptado, "
                "término más específico, término relacionado). No repitas variantes "
                "prácticamente idénticas entre sí. "
                "Responde EXCLUSIVAMENTE con un JSON de lista de strings, sin texto "
                'adicional ni markdown: ["variante 1", "variante 2", "variante 3", "variante 4"]'
            ),
            messages=[{"role": "user", "content": query}],
        )
        raw = message.content[0].text.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        variantes = json.loads(clean)
        if isinstance(variantes, list) and variantes:
            return [str(v).strip() for v in variantes[:n] if str(v).strip()]
    except Exception:
        pass
    return [query]


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


async def _fetch_pubmed_variant(query: str, max_results: int) -> list[dict]:
    """Ejecuta esearch + esummary para UNA variante de búsqueda puntual."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

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

        summary = await c.get(f"{base}/esummary.fcgi", params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }, headers=HEADERS)
        if summary.status_code != 200:
            return []
        result_data = summary.json().get("result", {})

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
        papers.append({
            "pmid": pmid,
            "title": item.get("title", ""),
            "authors": authors,
            "journal": item.get("source", ""),
            "year": item.get("pubdate", "")[:4],
            "doi": doi,
            "abstract": "",
            "open_access": doi is not None,
        })
    return papers


async def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """
    Busca papers en PubMed usando VARIAS variantes MeSH/sinónimos del término
    original (generadas por Claude vía generate_search_variants), en paralelo.
    Fusiona los resultados por PMID (sin duplicados), puntúa cada paper con
    score_paper() — el mismo sistema que usa el Radar de Literatura — y
    devuelve ordenado por score descendente, cortado a max_results.
    """
    variantes = await generate_search_variants(query, n=4)

    resultados_por_variante = await asyncio.gather(
        *(_fetch_pubmed_variant(v, max_results) for v in variantes)
    )

    fusion: dict[str, dict] = {}
    for lista in resultados_por_variante:
        for paper in lista:
            pmid = paper["pmid"]
            if pmid not in fusion:
                fusion[pmid] = paper

    for paper in fusion.values():
        paper["score"] = score_paper(paper)

    ranked = sorted(fusion.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:max_results]
                                
