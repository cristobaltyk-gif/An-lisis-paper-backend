"""
modules/document_generator.py — EvidenciaMed

Sintetiza un conjunto de papers ya analizados (mismo formato que devuelve
modules/analysis.analyze) en un único artículo de revisión clínica con
10 secciones fijas, listo para que el frontend lo traduzca a .docx.

La sección de Epidemiología internacional y nacional es la única que se
nutre de una búsqueda PubMed adicional (vía modules.search.search_pubmed),
independiente de los papers marcados por el usuario.
"""

import os
import json
import anthropic

from modules.search import search_pubmed

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DOCUMENT_KEY = os.getenv("DOCUMENT_KEY", "")

SYSTEM_PROMPT_DOCUMENT = """Eres un médico especialista en traumatología y ortopedia, experto en redacción de artículos de revisión clínica basados en evidencia.

Recibirás:
1. Una lista de papers ya analizados críticamente (cada uno con su PICO, nivel de evidencia Oxford, grado de recomendación, hallazgos clave y limitaciones).
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
  "clinica": "presentación clínica, síntomas y signos característicos según los papers analizados",
  "diagnostico": "criterios y método diagnóstico según los papers analizados",
  "diagnostico_diferencial": "principales diagnósticos diferenciales mencionados o clínicamente relevantes al cuadro",
  "examenes_complementarios": "estudios de laboratorio/imagenología relevantes según los papers analizados",
  "tratamientos": "opciones terapéuticas, con su nivel de evidencia y grado de recomendación según los papers analizados",
  "resultados": "resultados y desenlaces reportados en los papers analizados (tamaños de muestra, seguimiento, hallazgos cuantitativos)",
  "conclusiones": "síntesis final: qué dice la evidencia en conjunto, calidad global, y aplicabilidad clínica",
  "referencias": ["referencia 1 en formato autor-año-revista", "referencia 2", "..."]
}

Reglas estrictas:
- Cada afirmación con cifras o hallazgos debe poder rastrearse a alguno de los papers entregados.
- Si un apartado no tiene información suficiente en el material recibido, dilo explícitamente en ese campo (ej. "No se dispone de datos nacionales en los papers analizados") en vez de completar con conocimiento externo.
- La lista "referencias" debe incluir TODOS los papers usados (tanto los analizados críticamente como los de la búsqueda epidemiológica), en el mismo orden en que se citan en el texto."""


def _formatear_papers_analizados(papers: list[dict]) -> str:
    """Convierte la lista de papers ya analizados (formato de modules.analysis.analyze)
    en texto legible para el prompt."""
    bloques = []
    for i, p in enumerate(papers, 1):
        pico = p.get("pico", {})
        bloques.append(
            f"--- PAPER ANALIZADO {i} ---\n"
            f"Título: {p.get('titulo', '')}\n"
            f"Autores: {p.get('autores', '')}\n"
            f"Revista: {p.get('revista', '')}\n"
            f"DOI: {p.get('doi', '')}\n"
            f"Tipo de estudio: {p.get('tipo_estudio', '')}\n"
            f"Nivel de evidencia Oxford: {p.get('nivel_evidencia_oxford', '')} "
            f"({p.get('nivel_evidencia_descripcion', '')})\n"
            f"Grado de recomendación: {p.get('grado_recomendacion', '')}\n"
            f"Calidad GRADE: {p.get('calidad_grade', '')}\n"
            f"PICO — Población: {pico.get('poblacion', '')} | "
            f"Intervención: {pico.get('intervencion', '')} | "
            f"Comparador: {pico.get('comparador', '')} | "
            f"Outcome: {pico.get('outcome', '')}\n"
            f"Resumen ejecutivo: {p.get('resumen_ejecutivo', '')}\n"
            f"Hallazgos clave: {'; '.join(p.get('hallazgos_clave', []))}\n"
            f"Tamaño de muestra: {p.get('tamano_muestra', '')}\n"
            f"Seguimiento: {p.get('seguimiento', '')}\n"
            f"Limitaciones: {'; '.join(p.get('limitaciones', []))}\n"
            f"Aplicabilidad clínica: {p.get('aplicabilidad_clinica', '')}\n"
            f"Aplicabilidad Chile: {p.get('aplicabilidad_chile', '')}\n"
            f"Conclusión crítica: {p.get('conclusion_critica', '')}\n"
        )
    return "\n".join(bloques)


def _formatear_papers_epidemiologia(papers_epi: list[dict]) -> str:
    """Convierte los resultados crudos de search_pubmed (lista de metadatos,
    no analizados) en texto legible para el prompt."""
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


def build_document_content(papers: list[dict], papers_epi: list[dict], tema: str, autor: str) -> str:
    """Arma el contenido completo a enviar a Claude para la síntesis."""
    return (
        f"TEMA DEL DOCUMENTO: {tema}\n"
        f"AUTOR/INTERROGADOR RESPONSABLE: {autor}\n\n"
        f"===== PAPERS ANALIZADOS CRÍTICAMENTE (fuente principal para todas las secciones) =====\n\n"
        f"{_formatear_papers_analizados(papers)}\n\n"
        f"===== PAPERS DE BÚSQUEDA PUBMED — SOLO PARA LA SECCIÓN DE EPIDEMIOLOGÍA =====\n\n"
        f"{_formatear_papers_epidemiologia(papers_epi)}"
    )


async def generate_document(papers: list[dict], tema: str, autor: str = "") -> dict:
    """
    Orquesta la generación completa del documento:
    1. Busca papers de epidemiología en PubMed relacionados al tema.
    2. Arma el contenido de síntesis.
    3. Llama a Claude para obtener el JSON de las 10 secciones.
    4. Reintenta una vez si la respuesta llega truncada (mismo patrón que modules.analysis.analyze).
    """
    if not papers:
        raise ValueError("Se requiere al menos un paper analizado para generar el documento.")

    papers_epi = await buscar_epidemiologia(tema)
    content = build_document_content(papers, papers_epi, tema, autor or "No especificado")

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
  
