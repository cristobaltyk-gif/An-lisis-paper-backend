import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """Eres un experto en medicina basada en evidencia y metodología de investigación científica clínica.
Analiza el artículo científico y devuelve EXCLUSIVAMENTE un JSON válido con esta estructura exacta, sin texto adicional:

{
  "titulo": "título del paper",
  "autores": "autores principales",
  "revista": "nombre de la revista y año",
  "doi": "DOI del artículo",
  "tipo_estudio": "tipo de diseño metodológico",
  "nivel_evidencia_oxford": "1a/1b/2a/2b/3a/3b/4/5",
  "nivel_evidencia_descripcion": "descripción del nivel Oxford",
  "grado_recomendacion": "A/B/C/D",
  "calidad_grade": "Alta/Moderada/Baja/Muy baja",
  "pico": {
    "poblacion": "descripción de la población estudiada",
    "intervencion": "intervención principal",
    "comparador": "grupo control o comparador",
    "outcome": "desenlaces principales medidos"
  },
  "resumen_ejecutivo": "resumen en 3-4 oraciones claras del hallazgo principal",
  "hallazgos_clave": ["hallazgo 1", "hallazgo 2", "hallazgo 3"],
  "tamano_muestra": "N total y características",
  "seguimiento": "duración del seguimiento",
  "limitaciones": ["limitación 1", "limitación 2", "limitación 3"],
  "aplicabilidad_clinica": "cómo aplicar estos resultados en práctica clínica",
  "aplicabilidad_chile": "relevancia específica para el contexto chileno/latinoamericano",
  "conclusion_critica": "evaluación crítica honesta de fortalezas y debilidades",
  "puntuacion_calidad": 75,
  "categoria": "Terapéutica/Diagnóstico/Pronóstico/Etiología/Revisión/Meta-análisis/Guía Clínica"
}"""


def build_content(meta: dict, doi: str, fulltext: str | None) -> str:
    """Construye el contenido a enviar a Claude según disponibilidad de texto."""
    base = (
        f"DOI: {doi}\n"
        f"Título: {meta.get('title', '')}\n"
        f"Autores: {meta.get('authors', '')}\n"
        f"Revista: {meta.get('journal', '')} ({meta.get('year', '')})\n\n"
    )
    if fulltext and len(fulltext) > 500:
        return base + f"TEXTO COMPLETO:\n{fulltext}"
    elif meta.get("abstract"):
        return base + f"ABSTRACT:\n{meta['abstract']}\n\nNota: Solo se dispone del abstract."
    return base + "Nota: No hay abstract ni texto completo disponible. Analiza con los metadatos."


async def analyze(content: str) -> dict:
    """Llama a Claude y devuelve el JSON parseado."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    raw = message.content[0].text
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)
