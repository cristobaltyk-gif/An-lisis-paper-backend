"""
modules/document_generator.py — EvidenciaMed

Sintetiza un conjunto de papers CRUDOS (el mismo formato que devuelve
modules.search.search_pubmed: título, autores, revista, año, DOI, PMID)
en un único artículo de revisión clínica con 10 secciones fijas.

A diferencia del flujo de /analyze/doi (que hace análisis PICO/Oxford para
la app EvidenciaMed), este módulo NO pasa por ese análisis — es un pipeline
interno distinto, propio del módulo de documentos: obtiene el texto completo
o abstract de cada paper marcado (EN PARALELO, no uno por uno) y va directo
a la síntesis del documento.

La sección de Epidemiología internacional y nacional es la única que se
nutre de una búsqueda PubMed adicional (vía modules.search.search_pubmed),
independiente de los papers marcados por el interrogador.
"""

import os
import json
import asyncio
import anthropic

from modules.search import search_pubmed
from modules.downloader import get_fulltext

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DOCUMENT_KEY = os.getenv("DOCUMENT_KEY", "")

SYSTEM_PROMPT_DOCUMENT = """Eres un médico especialista en traumatología y ortopedia, experto en redacción de artículos de revisión clínica basados en evidencia.

Recibirás:
1. Una lista de papers marcados por un interrogador (con su texto completo cuando esté disponible, o solo abstract/metadatos si no lo está).
2. Resultados de una búsqueda PubMed adicional específica sobre epidemiología del tema (internacional y, si está disponible, de Chile/Latinoamérica).
3. El tema/título general del documento.

Tu tarea es sintetizar TODO ese material en un único artículo de revisión clínica, ordenado y coherente, usando EXCLUSIVAMENTE la información contenida en los papers entregados (no inventes datos, cifras ni referencias que no estén en el material recibido).

Devuelve EXCLUSIVAMENTE un JSON válido con esta estructura exacta, sin texto adicional ni markdown:

{
  "titulo": "título del artículo de revisión",
  "autor": "nombre del autor/interrogador responsable",
  "introduccion": "contexto general del tema, relevancia clínica, 2-3 párrafos",
  "epidemiologia": {
    "internacional": "epidemiología internacional basada en los papers de búsqueda PubMed entregados, citando cifras concretas",
    "nacional": "epidemiología en Chile/Latinoamérica si los papers de búsqueda la contienen; si no hay datos nacionales disponibles en el material entregado, indícalo explícitamente en vez de inventar cifras"
  },
  "clinica": "presentación clínica, síntomas y signos característicos según los papers entregados",
  "diagnostico": "criterios y método diagnóstico según los papers entregados",
  "diagnostico_diferencial": "principales diagnósticos diferenciales mencionados o clínicamente relevantes al cuadro",
  "examenes_complementarios": "estudios de laboratorio/imagenología relevantes según los papers entregados",
  "tratamientos": "opciones terapéuticas descritas en los papers entregados",
  "resultados": "resultados y desenlaces reportados en los papers entregados (tamaños de muestra, seguimiento, hallazgos cuantitativos)",
  "conclusiones": "síntesis final: qué dice el conjunto de papers entregados, y aplicabilidad clínica",
  "referencias": ["referencia 1 en formato autor-año-revista", "referencia 2", "..."]
}

Reglas estrictas:
- Cada afirmación con cifras o hallazgos debe poder rastrearse a alguno de los papers entregados.
- Si un paper solo tiene metadatos (sin texto completo ni abstract disponible), NO le atribuyas hallazgos específicos — menciónalo solo como referencia general o exclúyelo de afirmaciones puntuales.
- Si un apartado no tiene información suficiente en el material recibido, dilo explícitamente en ese campo (ej. "No se dispone de datos nacionales en los papers analizados") en vez de completar con conocimiento externo.
- La lista "referencias" debe incluir TODOS los papers marcados por el interrogador, en el mismo orden en que se citan en el texto."""


async def _obtener_texto_paper(paper: dict) -> str:
    """Intenta obtener texto completo del paper (por DOI o PMID);
    si no hay disponible, retorna solo un aviso de metadatos."""
    doi = paper.get("doi")
    pmid = paper.get("pmid")

    fulltext = None
    if doi:
        fulltext, _fuente = await get_fulltext(doi=doi)
    if not fulltext and pmid:
        fulltext, _fuente = await get_fulltext(pmid=pmid)

    if fulltext:
        return fulltext
    return "SOLO METADATOS DISPONIBLES — sin texto completo ni abstract accesible."


def _bloque_paper(p: dict, texto: str, i: int) -> str:
    return (
        f"--- PAPER MARCADO {i} ---\n"
        f"Título: {p.get('title', p.get('titulo', ''))}\n"
        f"Autores: {p.get('authors', p.get('autores', ''))}\n"
        f"Revista: {p.get('journal', p.get('revista', ''))} ({p.get('year', '')})\n"
        f"DOI: {p.get('doi', '')}\n"
        f"PMID: {p.get('pmid', '')}\n"
        f"Contenido:\n{texto}\n"
    )


async def _formatear_papers_marcados(papers: list[dict]) -> str:
    """Descarga el texto de TODOS los papers marcados en paralelo
    (asyncio.gather, no un for secuencial) y arma el bloque de texto."""
    textos = await asyncio.gather(*(_obtener_texto_paper(p) for p in papers))
    bloques = [_bloque_paper(p, texto, i) for i, (p, texto) in enumerate(zip(papers, textos), 1)]
    return "\n".join(bloques)


def _formatear_papers_epidemiologia(papers_epi: list[dict]) -> str:
    """Convierte los resultados crudos de search_pubmed en texto legible para el prompt
    (solo metadatos, no se descarga texto completo para esta sección)."""
    if not papers_epi:
        return "No se encontraron papers adicionales de epidemiología en PubMed para este tema."
    bloques = []
    for i, p in enumerate(papers_epi, 1):
        bloques.append(
            f"--- PAPER EPIDEMIOLOGÍA {i} ---\n"
            f"Título: {p.get('title', '')}\n"
            f"Autores: {p.get('authors', '')}\n"
            f"Revista: {p.get('journal', '')} ({p.get('year', '')})\n"
            f"PMID: {p.get('pmid', '')}\n"
            f"DOI: {p.get('doi', '')}\n"
        )
    return "\n".join(bloques)


async def buscar_epidemiologia(tema: str) -> list[dict]:
    """Busca en PubMed papers de epidemiología asociados al tema, reutilizando
    la misma función search_pubmed que usa el endpoint /search."""
    query = f"{tema} epidemiology incidence prevalence"
    resultados = await search_pubmed(query, max_results=8)
    return resultados or []


async def build_document_content(papers: list[dict], papers_epi: list[dict], tema: str, autor: str) -> str:
    """Arma el contenido completo a enviar a Claude para la síntesis.
    La descarga de texto de los papers y la búsqueda de epidemiología
    corren en paralelo entre sí también."""
    papers_texto, = await asyncio.gather(_formatear_papers_marcados(papers))
    return (
        f"TEMA DEL DOCUMENTO: {tema}\n"
        f"AUTOR/INTERROGADOR RESPONSABLE: {autor}\n\n"
        f"===== PAPERS MARCADOS POR EL INTERROGADOR (fuente principal para todas las secciones) =====\n\n"
        f"{papers_texto}\n\n"
        f"===== PAPERS DE BÚSQUEDA PUBMED — SOLO PARA LA SECCIÓN DE EPIDEMIOLOGÍA =====\n\n"
        f"{_formatear_papers_epidemiologia(papers_epi)}"
    )


async def generate_document(papers: list[dict], tema: str, autor: str = "") -> dict:
    """
    Orquesta la generación completa del documento:
    1. Busca papers de epidemiología en PubMed Y descarga el texto de todos
       los papers marcados EN PARALELO (no secuencial).
    2. Arma el contenido de síntesis.
    3. Llama a Claude para obtener el JSON de las 10 secciones.
    4. Reintenta una vez si la respuesta llega truncada.
    """
    if not papers:
        raise ValueError("Se requiere al menos un paper marcado para generar el documento.")

    papers_epi, papers_texto = await asyncio.gather(
        buscar_epidemiologia(tema),
        _formatear_papers_marcados(papers),
    )
    content = (
        f"TEMA DEL DOCUMENTO: {tema}\n"
        f"AUTOR/INTERROGADOR RESPONSABLE: {autor or 'No especificado'}\n\n"
        f"===== PAPERS MARCADOS POR EL INTERROGADOR (fuente principal para todas las secciones) =====\n\n"
        f"{papers_texto}\n\n"
        f"===== PAPERS DE BÚSQUEDA PUBMED — SOLO PARA LA SECCIÓN DE EPIDEMIOLOGÍA =====\n\n"
        f"{_formatear_papers_epidemiologia(papers_epi)}"
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT_DOCUMENT,
        messages=[{"role": "user", "content": content}],
    )
    raw = message.content[0].text
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        resultado = json.loads(clean)
    except json.JSONDecodeError:
        retry_content = (
            content
            + "\n\nIMPORTANTE: tu respuesta anterior se cortó por exceder el límite "
            + "de longitud. Responde de forma MÁS BREVE en cada sección de texto "
            + "(especialmente introduccion, clinica, tratamientos y resultados), "
            + "manteniendo el JSON completo y válido con TODAS las claves solicitadas."
        )
        message_retry = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT_DOCUMENT,
            messages=[{"role": "user", "content": retry_content}],
        )
        raw_retry = message_retry.content[0].text
        clean_retry = raw_retry.replace("```json", "").replace("```", "").strip()
        resultado = json.loads(clean_retry)

    resultado["tema"] = tema
    if autor:
        resultado["autor"] = autor
    return resultado


def verify_document_key(key: str) -> bool:
    """Verifica la clave fija del módulo de generación de documentos.
    Se usa desde main.py como dependencia del endpoint /generate/document."""
    return bool(DOCUMENT_KEY) and key == DOCUMENT_KEY
