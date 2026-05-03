import httpx
import fitz  # PyMuPDF
from typing import Optional

HEADERS = {"User-Agent": "EvidenciaMed/1.0 (mailto:contacto@cleversalud.cl)"}
EMAIL   = "contacto@cleversalud.cl"


async def _pdf_to_text(content: bytes) -> Optional[str]:
    """Extrae texto de bytes de PDF."""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text[:15000] if text.strip() else None
    except Exception:
        return None


async def from_unpaywall(doi: str) -> Optional[str]:
    """Descarga PDF open access desde Unpaywall."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}",
                headers=HEADERS
            )
            if r.status_code != 200:
                return None
            data = r.json()
            best = data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf") or best.get("url")
            if not pdf_url:
                return None
            pr = await c.get(pdf_url, follow_redirects=True, timeout=30)
            if pr.status_code == 200 and "pdf" in pr.headers.get("content-type", ""):
                return await _pdf_to_text(pr.content)
    except Exception:
        pass
    return None


async def from_pmc(doi: str = None, pmid: str = None) -> Optional[str]:
    """Descarga texto completo desde PubMed Central."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            # Convertir DOI o PMID a PMCID
            params = {"format": "json"}
            if doi:
                params["ids"] = doi
            elif pmid:
                params["ids"] = pmid
                params["idtype"] = "pmid"
            else:
                return None

            r = await c.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params=params, headers=HEADERS
            )
            if r.status_code != 200:
                return None
            records = r.json().get("records", [])
            if not records:
                return None
            pmcid = records[0].get("pmcid")
            if not pmcid:
                return None

            # Descargar texto
            fetch = await c.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pmc",
                    "id": pmcid.replace("PMC", ""),
                    "rettype": "text",
                    "retmode": "text",
                },
                headers=HEADERS
            )
            if fetch.status_code == 200 and len(fetch.text.strip()) > 200:
                return fetch.text.strip()[:15000]
    except Exception:
        pass
    return None


async def from_europepmc(doi: str) -> Optional[str]:
    """Intenta obtener texto desde Europe PMC — cubre más papers que PMC."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            # Buscar el paper por DOI
            r = await c.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": f"DOI:{doi}", "format": "json", "resultType": "core"},
                headers=HEADERS
            )
            if r.status_code != 200:
                return None
            results = r.json().get("resultList", {}).get("result", [])
            if not results:
                return None
            item = results[0]
            # Intentar obtener fulltext si está disponible
            pmcid = item.get("pmcid")
            if not pmcid:
                return None
            # Descargar XML de texto completo
            xml = await c.get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                headers=HEADERS, timeout=20
            )
            if xml.status_code != 200:
                return None
            # Extraer texto plano del XML (simple)
            import re
            text = re.sub(r'<[^>]+>', ' ', xml.text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:15000] if len(text) > 200 else None
    except Exception:
        pass
    return None


async def get_fulltext(doi: str = None, pmid: str = None) -> tuple[Optional[str], str]:
    """
    Estrategia completa de descarga en orden de preferencia:
    1. Unpaywall  → PDF open access
    2. PMC        → texto completo gratuito
    3. Europe PMC → cobertura adicional
    4. None       → solo abstract disponible

    Retorna (texto, fuente) donde fuente es:
    'unpaywall' | 'pmc' | 'europepmc' | 'abstract'
    """
    if doi:
        # 1. Unpaywall
        text = await from_unpaywall(doi)
        if text and len(text) > 500:
            return text, "unpaywall"

        # 2. PMC
        text = await from_pmc(doi=doi)
        if text and len(text) > 500:
            return text, "pmc"

        # 3. Europe PMC
        text = await from_europepmc(doi)
        if text and len(text) > 500:
            return text, "europepmc"

    elif pmid:
        # Solo PMC si tenemos PMID
        text = await from_pmc(pmid=pmid)
        if text and len(text) > 500:
            return text, "pmc"

    return None, "abstract"


def sciencedirect_url(doi: str) -> str:
    """URL directa al paper en ScienceDirect para acceso institucional."""
    return f"https://www.sciencedirect.com/science/article/pii/{doi.replace('/', '-')}"


def pubmed_url(pmid: str) -> str:
    """URL directa al paper en PubMed."""
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                                
