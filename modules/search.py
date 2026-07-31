import os
import re
import html as html_lib
import asyncio
import httpx
import fitz  # PyMuPDF
import anthropic
from typing import Optional
from modules.screener import score_paper

HEADERS = {"User-Agent": "EvidenciaMed/1.0 (mailto:contacto@cleversalud.cl)"}

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Credencial de SerpAPI (reemplaza a Google Custom Search, que quedó
# cerrada para proyectos nuevos — ver historial de la implementación).
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# PID de SciELO: S + ISSN (9 chars, ej. 0102-311X) + 13 dígitos de sufijo
SCIELO_PID_REGEX = re.compile(r"S\d{4}-\d{3}[0-9Xx]\d{13}")


def clean_doi(raw: str) -> str:
    """Normaliza cualquier formato de DOI."""
    doi = raw.strip()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


async def translate_to_english(query: str) -> str:
    """
    Traduce un término de búsqueda clínico a inglés para PubMed.
    Si el query ya está en inglés o falla la traducción, devuelve el original.
    """
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system=(
                "Eres un traductor especializado en terminología médica. "
                "Traduce el siguiente término de búsqueda al inglés, usando "
                "terminología médica/MeSH estándar (ej: 'prótesis de cadera' -> "
                "'hip arthroplasty', 'artritis reumatoide' -> 'rheumatoid arthritis'). "
                "Si ya está en inglés, devuélvelo igual. "
                "Responde EXCLUSIVAMENTE con el término traducido, sin comillas, "
                "sin explicaciones, sin texto adicional."
            ),
            messages=[{"role": "user", "content": query}],
        )
        translated = message.content[0].text.strip()
        return translated if translated else query
    except Exception:
        # Si falla la traducción (sin API key, error de red, etc.),
        # se sigue con el query original para no romper la búsqueda.
        return query


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
    Traduce automáticamente el término a inglés antes de buscar,
    ya que PubMed está indexado principalmente en inglés.
    Devuelve lista rankeada con score calculado.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    query_en = await translate_to_english(query)

    # 1. Buscar IDs
    async with httpx.AsyncClient(timeout=15) as c:
        search = await c.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed",
            "term": query_en,
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


# ---------------------------------------------------------------------------
# SciELO (colección Chile) — búsqueda por tema vía SerpAPI (Google real),
# restringida a site:scielo.cl, resolución de metadatos + fulltext vía
# la API oficial ArticleMeta (articlemeta.scielo.org).
# ---------------------------------------------------------------------------

async def _serpapi_site_search(query: str, site: str, max_results: int) -> list[str]:
    """Busca 'site:{site} {query}' vía SerpAPI (resultados reales de
    Google) y devuelve la lista de URLs de resultado. Si no hay
    credencial configurada (SERPAPI_KEY), devuelve lista vacía sin
    romper el resto del flujo."""
    if not SERPAPI_KEY:
        return []
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": f"site:{site} {query}",
                "num": min(max_results, 10),
                "api_key": SERPAPI_KEY,
            },
        )
        if r.status_code != 200:
            return []
        data = r.json()
        return [
            item.get("link", "")
            for item in data.get("organic_results", [])
            if item.get("link")
        ]


def _extraer_pid_scielo(url: str) -> Optional[str]:
    """Extrae el PID (código de artículo SciELO) de una URL de resultado."""
    match = SCIELO_PID_REGEX.search(url)
    return match.group(0) if match else None


async def _resolver_pid_scielo(pid: str) -> Optional[dict]:
    """Resuelve un PID contra la API ArticleMeta, colección Chile fija."""
    url = f"https://articlemeta.scielo.org/api/v1/article/?code={pid}&collection=chl"
    async with httpx.AsyncClient(timeout=20) as c:
        try:
            r = await c.get(url, headers=HEADERS)
        except httpx.RequestError:
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not data.get("code"):
            return None
        return data


async def _extraer_fulltext_html(html_url: str) -> str:
    """Descarga el HTML del artículo y extrae texto plano (sin bs4:
    quita script/style, quita tags, decodifica entidades)."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        try:
            r = await c.get(html_url, headers=HEADERS)
        except httpx.RequestError:
            return ""
        if r.status_code != 200:
            return ""
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", r.text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:15000]


async def _armar_paper_scielo(meta: dict) -> Optional[dict]:
    """Convierte un registro crudo de ArticleMeta (ISIS2JSON) al mismo
    shape que devuelve search_pubmed, agregando fulltext y fuente."""
    if not meta:
        return None

    article = meta.get("article", {})

    titulo = ""
    for t in article.get("v12", []):
        if t.get("l") == "es":
            titulo = t.get("_", "")
            break
    if not titulo and article.get("v12"):
        titulo = article["v12"][0].get("_", "")

    autores = ", ".join(
        f"{a.get('n','')} {a.get('s','')}".strip()
        for a in article.get("v10", [])[:4]
        if a.get("s")
    )

    revista = ""
    if article.get("v30"):
        revista = article["v30"][0].get("_", "")

    year = str(meta.get("publication_year", ""))[:4]
    doi = meta.get("doi", "")

    html_links = meta.get("fulltexts", {}).get("html", {})
    html_url = html_links.get("es") or html_links.get("pt") or html_links.get("en")
    if not html_url and html_links:
        html_url = next(iter(html_links.values()))

    fulltext = await _extraer_fulltext_html(html_url) if html_url else ""

    paper = {
        "pmid": None,
        "title": titulo,
        "authors": autores,
        "journal": revista,
        "year": year,
        "doi": doi,
        "abstract": "",
        "open_access": True,
        "fuente": "scielo",
        "fulltext": fulltext,
    }
    paper["score"] = score_paper(paper)
    return paper


async def search_scielo(query: str, max_results: int = 10) -> list[dict]:
    """
    Busca papers en SciELO (colección Chile) por tema, vía SerpAPI
    (resultados reales de Google) restringido a site:scielo.cl.

    Resuelve cada resultado por su PID contra la API oficial ArticleMeta
    y extrae el fulltext en HTML directo (SciELO es open access, no
    requiere Unpaywall/Elsevier). Devuelve lista rankeada con score,
    mismo shape que search_pubmed más los campos "fuente" y "fulltext".
    """
    urls = await _serpapi_site_search(query, "scielo.cl", max_results)

    pids = []
    vistos = set()
    for u in urls:
        pid = _extraer_pid_scielo(u)
        if pid and pid not in vistos:
            vistos.add(pid)
            pids.append(pid)

    if not pids:
        return []

    metas = await asyncio.gather(*(_resolver_pid_scielo(p) for p in pids))
    papers = await asyncio.gather(*(_armar_paper_scielo(m) for m in metas))
    papers = [p for p in papers if p]

    return sorted(papers, key=lambda x: x["score"], reverse=True)
